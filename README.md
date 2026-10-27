# MAPNet

Reference code and data for the paper

> **MAPNet: mean-anchored POD decoding for reconstructing LiFePO4 blade battery
> thermal fields from sparse measurements**

MAPNet reconstructs the internal temperature field of a 590 mm LiFePO4 blade
cell from the nine readings of embedded thermocouples. The field is written as

```
T_hat = b * 1 + mu_R + s_R * U16 @ a(s),      b = mean(s)
```

where `b` is the anchor (the mean of the nine readings), `mu_R` and `s_R` are
the training-set mean and RMS of the residual field, `U16` is a frozen
16-mode POD basis, and `a(s)` is the only learned part: a 10 -> 128 -> 16
residual MLP. The network therefore learns the *shape* of the field and never
its level, which is why 102,544 parameters suffice.

At a sensing noise of 0.05 K standard deviation the model reaches
**0.019193 K** RMSE on 2,000 held-out cases, 1.1 % below a parameter-matched
CNN (105,489) and 7.3 / 9.0 / 12.9 % below MLP, DeepONet and FNO.

## What is in this repository

```
src/          training, evaluation and figure scripts
data/         inputs: nine readings, POD basis, splits, 200-case demo
results/      archived metrics behind every table in the paper
checkpoints/  trained MAPNet weights (sigma = 0.05 K)
runs/         output directory, created on demand and not tracked
```

The training scripts are kept in the form in which the reported runs were
produced, including their historical internal name `pod16` for MAPNet.
`src/naming.py` maps internal names to the names used in the paper.

## Installation

```
python -m pip install -r requirements.txt
```

A CUDA-capable GPU is required for training; the demo runs on CPU as well.
The reported numbers were produced with Python 3.11, torch 2.13.0+cu130 in
float64 with TF32 disabled, on an RTX 5060 Ti (16 GB), i5-13400F, 32 GB RAM,
Windows 11.

## Quick start: inference demo (no training)

```
python src/demo_inference.py --plot
```

This loads `checkpoints/mapnet_noise_best.pt`, adds 0.05 K Gaussian noise with
the paper's seed to the nine readings of 200 held-out cases, reconstructs the
fields and reports the same metrics as the paper:

| quantity | value |
|---|---|
| parameters | 102,544 |
| RMSE (0.05 K noise, seed 31001) | 0.017879 K |
| MAE | 0.013937 K |
| peak MAE | 0.017071 K |
| RMSE (clean readings, `--sigma 0`) | 0.004860 K |

The demo subset is every tenth case of the 2,000-case test split, so its RMSE
differs slightly from the full-test value of 0.019193 K. Predictions and a
three-panel figure are written to `runs/demo/`.

## Reproducing the paper

1. **Field data.** `data/ydata.npy` (93,340 x 40 x 240, float32, 3.6 GB) is not
   stored here because it exceeds the GitHub file limit. Put it at
   `data/ydata.npy`, or point `MAPNET_DATA` at the directory that holds it:

   ```
   export MAPNET_DATA=/path/to/data      # must contain ydata.npy and sp.npy
   ```

   The nine readings for all 93,340 snapshots (`data/sp.npy`) *are* included,
   as are the POD basis and normalisation constants (`data/preprocessing.npz`)
   and the exact splits (`data/split_indices.npz`, `data/subset_indices.npz`).

2. **Train and test the five configurations** (MAPNet, CNN, DeepONet, FNO and
   the two-stage variant; 73,340 training snapshots, early stopping):

   ```
   python src/full_data_five_test.py --run
   ```

   About 2.2 h in total on the platform above, of which MAPNet takes 202 s
   (291 epochs, 0.69 s per epoch). Outputs go to `runs/full_data_five/`.

3. **MLP baseline** (107,241 parameters, same protocol):

   ```
   python src/full_data_mlp_test.py --run
   ```

4. **Tables and figures:**

   ```
   python src/interior_table.py --latex     # Table 2
   python src/noise_grid_test.py            # Tables 3 and 4 (sigma = 0.02-0.08 K)
   python src/per_case_inference.py         # per-case errors for Fig. 4
   python src/cost_benchmark.py             # Table 6 (latency, memory, MACs)
   python src/make_accuracy_figure.py       # Fig. 4
   python src/field_case_pick.py            # dump one case for Fig. 5
   python src/make_field_figure.py          # Fig. 5
   python src/make_architecture_figure.py   # Fig. 3
   ```

   `interior_table.py` and the figure scripts fall back to the archived
   `results/` when no training run is present, so Tables 2 and Figs. 3-5 can be
   regenerated without retraining. `cost_benchmark.py` needs the `done.json`
   logs written during training.

## Data files

| file | shape | content |
|---|---|---|
| `data/sp.npy` | 93,340 x 9, float32 | nine embedded readings, identical to the field values at the sensor locations |
| `data/preprocessing.npz` | - | residual mean, POD basis (9,600 x 16), scale, `b_mean`, `b_std`, training ids |
| `data/split_indices.npz` | 73,340 / 1,000 / 10,000 | training / validation / reserved test rows |
| `data/subset_indices.npz` | 2,000 | the test cases used in every table |
| `data/demo_fields.npy` | 200 x 40 x 240, float32 | demo fields (every tenth test case) |
| `data/demo_sensors.npy` | 200 x 9, float32 | demo readings |
| `data/demo_ids.npy` | 200, int64 | row index of each demo case in the full array |
| `data/protocol.json` | - | hyperparameters, seeds and protocol hash of the reported runs |

`ydata.npy` is available from the authors on request.

## Checkpoints

`checkpoints/mapnet_noise_best.pt` is the MAPNet checkpoint selected by the
noisy validation score at sigma = 0.05 K (epoch 251). It stores the network
weights together with the frozen POD basis and normalisation constants, so the
demo needs no other input than the nine readings.

## Results at a glance

Full-field RMSE in K on the 2,000 test cases (`results/test_metrics.csv`):

| model | parameters | clean | 0.02 K | 0.05 K | 0.08 K |
|---|---|---|---|---|---|
| MAPNet | 102,544 | 0.005110 | 0.008900 | 0.019193 | 0.030680 |
| CNN | 105,489 | 0.005116 | 0.008990 | 0.019404 | 0.030927 |
| MLP | 107,241 | 0.004572 | 0.009209 | 0.020600 | 0.032720 |
| DeepONet | 51,329 | 0.004484 | 0.009311 | 0.020926 | 0.033199 |
| FNO | 167,021 | 0.005292 | 0.009904 | 0.021665 | 0.034293 |

The 0.02 and 0.08 K columns are from `results/noise_grid_metrics.csv`; check
that file for the per-seed values and the exact seeds.

## Citation

If you use this code or data, please cite the paper above. The full reference
will be added once it is published.

## License

Code: MIT (see `LICENSE`). Data: released for research use with attribution to
the paper.
