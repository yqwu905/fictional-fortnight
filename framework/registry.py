OPS = {}
SELECTORS = {}


def register_op(name):
    def deco(cls_or_fn):
        OPS[name] = cls_or_fn
        return cls_or_fn
    return deco


def get_op(name):
    if name not in OPS:
        raise KeyError(f"unknown op type {name!r}. Registered ops: {sorted(OPS)}")
    return OPS[name]


def register_selector(name):
    def deco(fn):
        SELECTORS[name] = fn
        return fn
    return deco


def get_selector(name):
    if name not in SELECTORS:
        raise KeyError(f"unknown selector {name!r}. Registered selectors: {sorted(SELECTORS)}")
    return SELECTORS[name]
