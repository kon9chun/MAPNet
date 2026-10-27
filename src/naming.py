"""论文显示名映射：内部标识 -> 论文正式名称。

设计原则
--------
outputs/ 下的目录名、checkpoint 路径、metrics 的 config 列一律保留内部标识
（pod16 / cnn_small / ...），避免破坏已训练产物与既有结果的引用链。
仅在出图、出表、写论文时经本模块转为正式名称。

用法
----
    from naming import name, order_index
    ax.set_title(name("pod16_noise"))        # -> "MAPNet"
    df["model"] = df["config"].map(name)     # 整列替换
    labels = sorted(df["model"].unique(), key=order_index)
"""

DISPLAY = {
    # 本文模型
    "pod16": "MAPNet",
    "pod16_clean": "MAPNet",
    "pod16_noise": "MAPNet",
    "pod16_aug": "MAPNet",
    "pod16_staged": "MAPNet (two-stage)",
    # 基线
    "deeponet": "DeepONet",
    "deeponet_noise": "DeepONet",
    "fno2d": "FNO",
    "fno2d_noise": "FNO",
    "cnn_small": "CNN (0.11M)",
    "cnn_small_noise": "CNN (0.11M)",
    "cnn": "CNN (1.36M)",
    "mlp_small": "MLP (0.11M)",
    "mlp": "MLP (1.27M)",
}

ORDER = [
    "MAPNet",
    "MAPNet (two-stage)",
    "DeepONet",
    "FNO",
    "CNN (0.11M)",
    "CNN (1.36M)",
    "MLP (0.11M)",
    "MLP (1.27M)",
]

_UNKNOWN_SUFFIXES = ("_noise", "_clean", "_aug", "_staged")


def name(key):
    """内部标识 -> 论文显示名。未知标识回退到去后缀的原串，不抛异常。"""
    if key is None:
        return ""
    k = str(key).strip().lstrip("\ufeff")
    if k in DISPLAY:
        return DISPLAY[k]
    base = k
    for suf in _UNKNOWN_SUFFIXES:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return DISPLAY.get(base, k)


def order_index(label):
    """用于图例/表格排序；未收录的排在最后。"""
    try:
        return ORDER.index(label)
    except ValueError:
        return len(ORDER)
