"""全量协议补跑 mlp_small（107,241 参数），补齐表 1 的第五行。

背景：CNN / DeepONet / FNO / MAPNet 已有 73,340 样本全量结果
（outputs/full_data_five_20261022），但 MLP 只有 8,192 子集口径
（outputs/low_noise_param_matched_20261022），两者不能放进同一张表。
本脚本复用 full_data_five_test.py 的全部协议常量与训练/测试流程，
只把配置清单换成单一 mlp_small，输出到独立目录，
以免改动已完成的全量结果的 protocol.json / script_sha256。

结构定义来自 low_noise_param_matched_test.Net：
    10 -> 96 -> 96 -> 9 -> 9600, SiLU, 107,241 参数
即"可学习 rank-9 线性解码"，与 MAPNet 的固定 16 维 POD 基底形成对照。

运行
    python -B outputs/full_data_mlp_test.py --run
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402
import full_data_five_test as f5  # noqa: E402

# 独立输出目录：不动全量五配置的产物
f5.OUT = paths.RUNS / 'full_data_mlp'
f5.ORDER = ('mlp_small_noise',)
f5.KIND['mlp_small_noise'] = 'mlp_small'
f5.LABEL['mlp_small_noise'] = 'MLP(107K) sigma0.05'

if __name__ == '__main__':
    f5.main()
