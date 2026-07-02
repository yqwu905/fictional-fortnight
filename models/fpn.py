# Put the FPN model code provided in the prompt here unchanged.
# Do not modify the model structure in this file.
#
# Expected config target:
#   projects.flare_seg.models.fpn.FPN
#
# Important:
# The code pasted in the prompt references self.bn1 / self.bn2 in FPN.forward(),
# but the shown __init__ only defines self.gn1 / self.gn2. Because the requirement
# says not to modify the model structure, this project does not patch that model
# code automatically. Paste your actual complete model implementation here.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .repvit_m import repvit_m0_9_channel_mtl3_no_se_ReLU_groupconv_equalflops
from .repvit_m import RepViTBlock
from .repvit_m import Conv2d_BN as ConvBN
import math

class deconv_eq_bilinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, padding_mode='zeros', upscale_factor=2):
        super(deconv_eq_bilinear, self).__init__()

        self.upscale_factor = upscale_factor
        self.stride = upscale_factor
        self.kernel_size = upscale_factor * 2
        self.out_padding = 0
        self.padding = int((self.kernel_size - self.stride + self.out_padding)/2)
        self.per_group_ch = 16

        if in_channels < self.per_group_ch:
            self.per_group_ch = in_channels
        if upscale_factor == 3:
            self.kernel_size = 5
            self.padding = 1

        self.groups = math.floor(in_channels / self.per_group_ch)
        self.deconv_up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding, output_padding=self.out_padding, groups=self.groups, bias=bias,
            padding_mode=padding_mode)

        # 放开deconv参数训练
        # self.weights_init()
        # print('deconv_eq_bilinear up x %d weights_init is fixed' % upscale_factor)

    def forward(self, inp):
        output = self.deconv_up(inp)
        return output

class FPNAdvance_f4(nn.Module):
    def __init__(self, num_classes, backbone='repvit_m', down_sample=6, 
                 channel0=160, channel1=128, channel2=96, channel3=64,
                 pretrained=True, export_onnx=False):
        super(FPNAdvance_f4, self).__init__()
        
        self.num_classes = num_classes
        self.export_onnx = export_onnx
        self.down_sample = down_sample
        
        # Backbone
        self.backbone = repvit_m0_9_channel_mtl3_no_se_ReLU_groupconv_equalflops()
        
        # FPN layers
        self.fpn_layers = self._build_fpn_layers(channel0, channel1, channel2, channel3)
        
        # Semantic branches
        self.semantic_branches = self._build_semantic_branches(channel3)
        
        # Convolutional layers
        self.conv_layers = self._build_conv_layers(channel3)
        
        # Upsampling layers
        self.upsampling_layers = self._build_upsampling_layers(channel3, num_classes)
        
        # conv and classification
        self.classification_head = self._build_classification_head(channel3, num_classes)
        print(self)

    def _build_fpn_layers(self, channel0, channel1, channel2, channel3):
        """Build FPN (Feature Pyramid Network) layers"""
        return nn.ModuleDict({
            'top_layer': nn.Conv2d(channel0, channel3, kernel_size=1, stride=1, padding=0),
            'lateral_layer1': nn.Conv2d(channel1, channel3, kernel_size=1, stride=1, padding=0),
            'lateral_layer2': nn.Conv2d(channel2, channel3, kernel_size=1, stride=1, padding=0),
            'lateral_layer3': nn.Conv2d(channel3, channel3, kernel_size=1, stride=1, padding=0),
        })

    def _build_semantic_branches(self, channel3):
        """Build semantic segmentation branches"""
        branches = nn.ModuleList()
        for _ in range(4):
            branch = nn.Sequential(
                RepViTBlock(channel3, channel3 * 2, channel3, 3, 1, 0, 1),
                ConvBN(a=channel3, b=channel3 // 2, ks=1, stride=1, pad=0)
            )
            branches.append(branch)
        return branches

    def _build_conv_layers(self, channel3):
        """Build convolutional layers"""
        return nn.ModuleDict({
            'convbn1': ConvBN(a=channel3, b=channel3, ks=3, stride=1, pad=1),
            'convbn2': ConvBN(a=channel3, b=channel3, ks=3, stride=1, pad=1),
            'convbn4': ConvBN(a=channel3, b=channel3, ks=3, stride=1, pad=1),
        })

    def _build_upsampling_layers(self, channel3, num_classes):
        """Build upsampling layers"""
        return nn.ModuleDict({
            # Main feature upsampling
            'up_p5_1': deconv_eq_bilinear(in_channels=channel3, out_channels=channel3, upscale_factor=2),
            'up_p5_2': deconv_eq_bilinear(in_channels=channel3, out_channels=channel3, upscale_factor=2),
            'up_p5_3': deconv_eq_bilinear(in_channels=channel3, out_channels=channel3, upscale_factor=2),
            'up_s5_to_half': deconv_eq_bilinear(in_channels=channel3//2, out_channels=channel3 // 2, upscale_factor=2),
            
            'up_p4_1': deconv_eq_bilinear(in_channels=channel3, out_channels=channel3, upscale_factor=2),
            'up_p4_2': deconv_eq_bilinear(in_channels=channel3, out_channels=channel3, upscale_factor=2),
            'up_s4_to_half': deconv_eq_bilinear(in_channels=channel3//2, out_channels=channel3 // 2, upscale_factor=2),
            
            'up_p3_1': deconv_eq_bilinear(in_channels=channel3, out_channels=channel3, upscale_factor=2),
            'up_s3_to_half': deconv_eq_bilinear(in_channels=channel3//2, out_channels=channel3 // 2, upscale_factor=2),

            'comb_to_half': deconv_eq_bilinear(in_channels=channel3 // 2, out_channels=channel3 // 2, upscale_factor=2),
            
            # Final upsampling
            'final_upsample': deconv_eq_bilinear(in_channels=num_classes, out_channels=num_classes, upscale_factor=3),
        })

    def _build_classification_head(self, channel3, num_classes):
        """Build final classification head"""
        return nn.Sequential(
            ConvBN(channel3 // 2, channel3 // 2, ks=3, stride=1, pad=1),
            ConvBN(channel3 // 2, channel3 // 2, ks=3, stride=1, pad=1),
            ConvBN(channel3 // 2, num_classes, ks=1, stride=1, pad=0)
        )

    def _fpn_forward(self, features):
        """Forward pass through FPN layers"""
        c1, c2, c3, c4, c5 = features
        
        # Top-down pathway
        p5 = self.fpn_layers['top_layer'](c5)  # 64, size/32
        p4 = self._add_upsampled_features(p5, self.fpn_layers['lateral_layer1'](c4), 'up_p5_1')  # 64, size/16
        p3 = self._add_upsampled_features(p4, self.fpn_layers['lateral_layer2'](c3), 'up_p4_1')  # 64, size/8
        p2 = self._add_upsampled_features(p3, self.fpn_layers['lateral_layer3'](c2), 'up_p3_1')  # 64, size/4
        
        return p2, p3, p4, p5

    def _add_upsampled_features(self, x, y, upsample_key):
        """Upsample x and add with y"""
        upsampled_x = self.upsampling_layers[upsample_key](x)
        return upsampled_x + y

    def _semantic_processing(self, p2, p3, p4, p5):
        """Process features through semantic branches"""
        # Process P5: 64->64->64->32
        s5 = self.upsampling_layers['up_p5_2'](F.leaky_relu(self.conv_layers['convbn1'](p5)))
        s5 = self.upsampling_layers['up_p5_3'](F.leaky_relu(self.conv_layers['convbn2'](s5)))
        s5 = self.upsampling_layers['up_s5_to_half'](F.leaky_relu(self.semantic_branches[0](s5)))

        # Process P4: 64->64->32
        s4 = self.upsampling_layers['up_p4_2'](F.leaky_relu(self.conv_layers['convbn4'](p4)))
        s4 = self.upsampling_layers['up_s4_to_half'](F.leaky_relu(self.semantic_branches[1](s4)))

        # Process P3: 64->32
        s3 = self.upsampling_layers['up_s3_to_half'](F.leaky_relu(self.semantic_branches[2](p3)))

        # Process P2: 64->32
        s2 = F.leaky_relu(self.semantic_branches[3](p2))

        return s2, s3, s4, s5

    def forward(self, x):
        # 取4层特征
        features = self.backbone(x)  # [c1, c2, c3, c4, c5] with channels [32, 64, 96, 128, 160]
        
        # FPN过程得到4层特征
        p2, p3, p4, p5 = self._fpn_forward(features)
        
        # 各特征层逐步上采样到浅层特征尺寸
        s2, s3, s4, s5 = self._semantic_processing(p2, p3, p4, p5)
        
        # 各特征层叠加
        combined_features = s2 + s3 + s4 + s5
        
        # 卷积平滑
        combined_features = self.classification_head[0](combined_features)
        # 上采样
        output = self.upsampling_layers['comb_to_half'](combined_features)
        
        # 卷积平滑
        output = self.classification_head[1](output)
        # 分类得到各类别得分
        output = self.classification_head[2](output)
        
        # 上采样到原始尺寸
        output = self.upsampling_layers['final_upsample'](output)
        
        # 到处模型时加入后处理
        if self.export_onnx:
            output = output.argmax(dim=1)
        
        return output


if __name__ == "__main__":
    from torch.nn import CrossEntropyLoss
    from ptflops import get_model_complexity_info
    import sys

    net = FPNAdvance_f4(6).cuda()

    inp = (2 * torch.rand(1, 3, 1536, 1536) - 1).cuda()

    size_input = (3, 1536, 1536)
    ost = sys.stdout
    flops, params = get_model_complexity_info(net, size_input,
                                              as_strings=True,
                                              print_per_layer_stat=False,
                                              ost=ost)

    print('macs: ', flops)
    print('{:<30}  {:<8}'.format('Total net Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Total net Number of parameters: ', params))

    model = FPNAdvance_f4(num_classes=6)

    input = torch.rand(1,3,1536,1536)
    output = model(input)
    print(output.size())