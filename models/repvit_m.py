import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from sympy.strategies.branch import identity

sys.path.append(str(Path(__file__).parent.parent))


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channels = channels
        self.reduction = max(1, channels // reduction_ratio)  # 确保至少减少到1

        # 结构定义
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                     # Squeeze: [B, C, H, W] -> [B, C, 1, 1]
            nn.Conv2d(channels, self.reduction, 1),      # Excitation: FC1 (降维)
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduction, channels, 1),      # Excitation: FC2 (恢复维度)
            nn.Sigmoid()                                # 输出 [0,1] 的通道权重
        )

    def forward(self, x):
        weights = self.se(x)  # [B, C, 1, 1]
        return x * weights    # 特征图按通道加权


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


from timm.models.layers import SqueezeExcite

import torch


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.c = torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(b)
        self._fused = False
        self.fuse_conv = None
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)  # 权重初始化为指定值bn_weight_init
        torch.nn.init.constant_(self.bn.bias, 0)  # 偏置初始化为0

    def forward(self, x):
        if not self._fused:
            return self.bn(self.c(x))
        else:
            return self.fuse_conv(x)  # 融合后直接使用conv

    @torch.no_grad()
    def fuse(self, inplace=False):
        '''
        fuse()方法通过数学等价变换，将卷积和BN的线性运算合并为单个卷积操作
        权重融合：W_fused = W_conv * (γ / sqrt(σ² + ε))
        偏置融合：b_fused = β - (μ * γ / sqrt(σ² + ε))
        :return:
        '''
        if self._fused:
            return self if inplace else copy.deepcopy(self.conv)
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        # 创建融合后的新的卷积层
        self.fuse_conv = torch.nn.Conv2d(
            w.size(1) * self.c.groups,  # 输入通道数（考虑分组卷积）
            w.size(0),  # 输出通道数
            w.shape[2:],  # 卷积核大小
            stride=self.c.stride,  # 保持原步长
            padding=self.c.padding,  # 保持原填充
            dilation=self.c.dilation,  # 保持原空洞率
            groups=self.c.groups,  # 保持原分组数
            device=c.weight.device  # 保持原设备
        )
        self.fuse_conv.weight.data.copy_(w)
        self.fuse_conv.bias.data.copy_(b)
        if inplace:
            self.bn = None
            self.conv = None
            self._fused = True
            return self
        return self.fuse_conv



class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop
        self._fused = False

    def forward(self, x):
        if not self._fused:
            if self.training and self.drop > 0:
                return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                                  device=x.device).ge_(self.drop).div(1 - self.drop).detach()
            else:
                return x + self.m(x)
        else:
            return self.m(x)

    @torch.no_grad()
    def fuse(self, inplace=False):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            if m.weight.shape[1] > 1:
                identity = torch.zeros(m.weight.shape[0], m.weight.shape[1], 1, 1, device=m.weight.device)
                for i in range(m.weight.shape[0]):
                    identity[i, i % self.group_channels] = 1
            else:
                identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)

            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            if inplace:
                # 安全替换
                self.m = m
                self._fused = True  # 更新融合状态
                return self
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            if m.weight.shape[1] > 1:
                identity = torch.zeros(m.weight.shape[0], m.weight.shape[1], 1, 1, device=m.weight.device)
                for i in range(m.weight.shape[0]):
                    identity[i, i % self.group_channels] = 1
            else:
                identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            if inplace:
                # 安全替换
                self.m = m
                self._fused = True  # 更新融合状态
                return self
            return m
        return self



class RepVGGDW(torch.nn.Module):
    def __init__(self, ed, group_channels=1) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed // group_channels)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed // group_channels)
        self.bn = torch.nn.BatchNorm2d(ed)
        self._fused = False  # 添加融合状态标志
        self.group_channels = group_channels

    def forward(self, x):
        if not self._fused:
            return self.bn((self.conv(x) + self.conv1(x)) + x)
        else:
            return self.conv(x)  # 融合后直接使用conv

    @torch.no_grad()  # 关键修饰器，禁用梯度计算
    def fuse(self, inplace=False):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        if conv1_w.shape[1] > 1:
            pattern = torch.zeros(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device)
            for i in range(conv1_w.shape[0]):
                pattern[i, i % self.group_channels] = 1
            identity = torch.nn.functional.pad(pattern,[1, 1, 1, 1])
        else:
            identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        if inplace:
            # 安全替换
            self.conv = conv
            self.conv1 = None
            self.bn = None
            self._fused = True  # 更新融合状态
            return self
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        group_channel = 32
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            # token_mixer 负责空间信息交互
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp // group_channel),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            # channel_mixer 负责通道信息交互
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.LeakyReLU() if use_hs else nn.ReLU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp, group_channel),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.LeakyReLU() if use_hs else nn.ReLU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


from timm.models.vision_transformer import trunc_normal_


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self, inplace=False):
        bn, l = self._modules.values()

        # 计算融合后的权重和偏置
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        w_fused = l.weight * w[None, :]  # 缩放权重

        if l.bias is None:
            b_fused = b @ l.weight.T  # 无原始偏置时的融合
        else:
            b_fused = (l.weight @ b[:, None]).view(-1) + l.bias  # 合并原始偏置

        # 创建融合后的Linear层
        fused_linear = torch.nn.Linear(
            w_fused.size(1),
            w_fused.size(0),
            bias=True,  # 强制启用偏置（因BN有偏置项）
            device=l.weight.device
        )
        fused_linear.weight.data.copy_(w_fused)
        fused_linear.bias.data.copy_(b_fused)

        if inplace:
            # 原地替换：清空当前模块并转换为普通Linear
            self.__dict__.clear()
            self.__dict__.update(fused_linear.__dict__)
            return self
        return fused_linear


#
class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier


class RepViT(nn.Module):
    def __init__(self, cfgs, num_classes=1000, distillation=False, convert_to_onnx=False, convert_to_onnx_for_omc=False, in_features=["dark3", "dark4", "dark5"], convert_to_onnx_only_backbone=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.in_features = in_features
        self.feature_layers = [0]     # 不同尺寸的特征图所在层数
        self.convert_to_onnx_only_backbone = convert_to_onnx_only_backbone

        # building first layer
        input_channel = self.cfgs[0][2]

        patch_embed1 = torch.nn.Sequential(Conv2d_BN(4 if (convert_to_onnx and not convert_to_onnx_for_omc) else 3, input_channel // 2, 3, 3, 1), torch.nn.ReLU())
        patch_embed2 = Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1)
        layers = [patch_embed1, patch_embed2]
        # building inverted residual blocks
        block = RepViTBlock
        index = 0
        for k, t, c, use_se, use_hs, s in self.cfgs:
            index += 1
            if s == 2:
                self.feature_layers.append(index)
            output_channel = make_divisible(c, 8)
            exp_size = make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self.feature_layers.append(len(self.features) - 1)
        # self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 624, 624))]

    def forward(self, x):
        # 避免输入出现在控制流算子
        features_names = ["stem", "dark2", "dark3", "dark4", "dark5"]
        features = [None, None, None, None, None]
        for i, f in enumerate(self.features):
            x = f(x)
            if i in self.feature_layers:
                features[self.feature_layers.index(i)] = x
        if self.convert_to_onnx_only_backbone:
            features_output = []
            for f in self.in_features:
                features_output.append(features[features_names.index(f)])
            return features_output
        return features


class SEAM(nn.Module):
    def __init__(self, c1, c2, n, reduction=16):
        super(SEAM, self).__init__()
        if c1 != c2:
            c2 = c1
        self.DCovN = nn.Sequential(
            # nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, groups=c1),
            # nn.GELU(),
            # nn.BatchNorm2d(c2),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(c2)
            ) for i in range(n)]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

def DcovN(c1, c2, depth, kernel_size=3, patch_size=3):
    dcovn = nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.LeakyReLU(),
        nn.BatchNorm2d(c2),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(c2)
            )),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(c2)
        ) for i in range(depth)]
    )
    return dcovn

class MultiSEAM(nn.Module):
    def __init__(self, c1, c2, depth, kernel_size=3, patch_size=[3, 5, 7], reduction=16):
        super(MultiSEAM, self).__init__()
        if c1 != c2:
            c2 = c1
        self.DCovN0 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
        self.DCovN1 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
        self.DCovN2 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y0 = self.DCovN0(x)
        y1 = self.DCovN1(x)
        y2 = self.DCovN2(x)
        y0 = self.avg_pool(y0).view(b, c)
        y1 = self.avg_pool(y1).view(b, c)
        y2 = self.avg_pool(y2).view(b, c)
        y4 = self.avg_pool(x).view(b, c)
        y = (y0 + y1 + y2 + y4) / 4
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)


def repvit_m0_9_channel_mtl3_no_se_ReLU_groupconv_equalflops(num_classes=1000, distillation=False, reparam=False,
                                                             convert_to_onnx=False, convert_to_onnx_for_omc=False,
                                                             in_features=["dark3", "dark4", "dark5"],
                                                             convert_to_onnx_only_backbone=False):
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  96, 0, 0, 2],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2, 128, 0, 0, 2],
        [3,   2, 128, 0, 0, 1],
        [3,   2, 128, 0, 0, 1],
        [3,   2, 128, 0, 0, 1],
        [3,   2, 128, 0, 0, 1],
        [3,   2, 128, 0, 0, 1],
        [3,   2, 128, 0, 0, 1],
        [3,   2, 128, 0, 0, 1],
        [3,   2, 128, 0, 0, 1],
        [3,   2, 128, 0, 0, 1],
        [3,   2, 128, 0, 0, 1],
        [3,   2, 160, 0, 0, 2],
        [3,   2, 160, 0, 0, 1],
        [3,   2, 160, 0, 0, 1],
        [3,   2, 160, 0, 0, 1],
        [3,   2, 160, 0, 0, 1]
    ]
    model = RepViT(cfgs, num_classes=num_classes, distillation=distillation, convert_to_onnx=convert_to_onnx,
                  convert_to_onnx_for_omc=convert_to_onnx_for_omc, in_features=in_features,
                  convert_to_onnx_only_backbone=convert_to_onnx_only_backbone)
    if reparam:
        return fuse_model(model)
    return model


try:
    from thop import profile  # 用于计算FLOPs和参数数量的工具库
except ImportError:
    profile = None


def calculate_model_flops(model, input_size=(3, 224, 224)):
    """
    计算模型的FLOPs和参数数量

    参数:
        model: 要计算的PyTorch模型
        input_size: 输入图像的尺寸 (通道数, 高度, 宽度)

    返回:
        flops: 模型的FLOPs数量
        params: 模型的参数数量
    """
    # 创建一个随机输入张量，用于模拟输入数据
    input_tensor = torch.randn(1, *input_size)

    # 使用thop库的profile函数计算FLOPs和参数数量
    if profile is None:
        raise ImportError("calculate_model_flops requires thop. Install with: pip install thop")

    flops, params = profile(model, inputs=(input_tensor,))

    # 转换单位（从FLOPs转换为GigaFLOPs）
    flops_giga = flops / 1e9
    params_million = params / 1e6

    return flops_giga, params_million

def fuse_model(model):
    """
    遍历模型的所有模块，对支持重参数化的模块进行融合
    :param model: 待融合的模型
    :return: 融合后的模型
    """
    # 获取模型的所有子模块
    for module in model.children():
        # 如果模块本身有fuse方法，直接调用
        if hasattr(module, 'fuse'):
            module.fuse(inplace=True)
        # 递归处理子模块
        elif isinstance(module, torch.nn.Module):
            fuse_model(module)
    return model


def print_model_stats(model_name, model_func):
    """
    打印指定模型的统计信息
    """
    # 初始化模型
    model = model_func(pretrained=False, num_classes=1000)
    model.eval()  # 设置为评估模式
    # 控制是否重参数化
    model = fuse_model(model)

    # 计算FLOPs和参数数量
    flops, params = calculate_model_flops(model, input_size=(3, 624, 624))

    # 打印结果
    print(f"模型配置: {model_name}")
    print(f"FLOPs: {flops:.2f} G")
    print(f"参数数量: {params:.2f} M")
    print("------------------------")


if __name__ == "__main__":
    # 确保导入了所有的RepViT模型配置
    # 这里假设前面定义的RepViT相关类和函数都已导入

    # 统计各个配置的模型计算量
    print("RepViT各配置模型计算量统计:")
    print("------------------------")
    # print_model_stats("repvit_m0_6", repvit_m0_6)
    # print_model_stats("repvit_m0_9", repvit_m0_9)
    # print_model_stats("repvit_m1_0", repvit_m1_0)
    # print_model_stats("repvit_m1_1", repvit_m1_1)
    # print_model_stats("repvit_m1_5", repvit_m1_5)
    # print_model_stats("repvit_m2_3", repvit_m2_3)
    model = repvit_m0_9(pretrained=False, num_classes=1000)
    x = torch.randn(1, 3, 624, 624)
    out = model(x)
    for tensorx in out:
        print(f"Final output shape: {tensorx.shape}")

