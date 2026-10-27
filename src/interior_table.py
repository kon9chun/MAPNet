"""生成论文表 1（内部 9 测点口径）的 LaTeX 行，并给出选模规则交叉核对。

数据来源
    outputs/full_data_five_20261022/summary.json  MAPNet / CNN / DeepONet / FNO
    outputs/full_data_mlp_20261023/summary.json   MLP（107,241 参数，补跑）

口径约定
    表 1 只报**一个检查点**：按 sigma=0.05 含噪验证误差选出的那个（selection='noise'），
    clean 与 sigma=0.05 两列都来自同一检查点，避免"不同列取不同选模规则"。
    summary 里另存有按干净验证误差选模的分支（selection='clean'），
    两者差异在本脚本末尾一并打印，供正文"选模规则不影响排序"的论断引用。

运行
    python -B outputs/interior_table.py            # 表格 + 交叉核对
    python -B outputs/interior_table.py --latex    # 只打印 LaTeX 行
"""
import argparse
import json
import pathlib

import paths

ROOT = pathlib.Path(__file__).resolve().parent
MAIN = paths.RUNS / 'full_data_five' / 'summary.json'
if not MAIN.exists():                      # fall back to the archived results
    MAIN = paths.RESULTS / 'summary.json'
MLP = paths.RUNS / 'full_data_mlp' / 'summary.json'
if not MLP.exists():
    MLP = paths.RESULTS / 'mlp_summary.json'

# (summary 里的配置名, 论文显示名, 顺序)
ORDER = [('pod16_noise', 'MAPNet'), ('cnn_small_noise', 'CNN'),
         ('deeponet_noise', 'DeepONet'), ('fno2d_noise', 'FNO')]
SELECTION = 'noise'


def load():
    table, params = {}, {}
    for path, entries in ((MAIN, ORDER), (MLP, [('mlp_small_noise', 'MLP')])):
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding='utf-8'))
        for cfg, label in entries:
            if cfg not in d['table']:
                continue
            sel = SELECTION if SELECTION in d['table'][cfg] else 'clean'
            table[label] = d['table'][cfg][sel]
            table[label + '|other'] = d['table'][cfg][
                'clean' if sel == 'noise' else 'noise']
            params[label] = d['training'][cfg]['stages'][-1]['parameters']
    return table, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latex', action='store_true')
    args = ap.parse_args()
    table, params = load()
    base = table['MAPNet'][f'noise_0.05']['RMSE_K']
    rows = [('MAPNet', name) for name in table if not name.endswith('|other')]
    order = ['MAPNet', 'CNN', 'DeepONet', 'FNO', 'MLP']
    if args.latex:
        for name in order:
            if name not in table:
                continue
            v = table[name]
            rel = '---' if name == 'MAPNet' else \
                f'+{(v[f"noise_0.05"]["RMSE_K"] / base - 1) * 100:.1f}\\%'
            print(f'{name:<8} & {params[name]:>9,} & {v["clean"]["RMSE_K"]:.6f} & '
                  f'{v[f"noise_0.05"]["RMSE_K"]:.6f} & ${rel}$ \\\\'.replace(',', '{,}'))
        return
    print(f'{"model":<10}{"params":>9}{"clean":>11}{"s=0.05":>11}{"rel":>9}')
    for name in order:
        if name not in table:
            continue
        v, o = table[name], table[name + '|other']
        rel = '--' if name == 'MAPNet' else f'{(v[f"noise_0.05"]["RMSE_K"] / base - 1) * 100:+.1f}%'
        print(f'{name:<10}{params[name]:>9,}{v["clean"]["RMSE_K"]:>11.6f}'
              f'{v[f"noise_0.05"]["RMSE_K"]:>11.6f}{rel:>9}')
        print(f'{"":<10}{"  (另一选模)":<9}{o["clean"]["RMSE_K"]:>11.6f}'
              f'{o[f"noise_0.05"]["RMSE_K"]:>11.6f}')
    print('\n选模规则敏感性（|noise-best − clean-best| / clean-best）：')
    for name in order:
        if name not in table:
            continue
        a, b = table[name], table[name + '|other']
        d05 = abs(a['noise_0.05']['RMSE_K'] - b['noise_0.05']['RMSE_K']) / b['noise_0.05']['RMSE_K']
        dc = abs(a['clean']['RMSE_K'] - b['clean']['RMSE_K']) / b['clean']['RMSE_K']
        print(f'  {name:<9} sigma=0.05 {d05 * 100:.2f}%   clean {dc * 100:.2f}%')


if __name__ == '__main__':
    main()
