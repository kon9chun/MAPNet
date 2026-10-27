"""Repository-wide path constants.

Layout (after cloning):

    MAPNet/
      src/          these scripts
      data/         inputs (sensor readings, POD basis, splits, demo fields)
      results/      archived metrics reported in the paper
      checkpoints/  trained MAPNet weights
      runs/         output directory, created on demand, not tracked by git

Every path can be relocated with an environment variable, which is handy when
the 3.6 GB field array lives on another disk:

    MAPNET_DATA  directory holding ydata.npy / sp.npy and the split files
    MAPNET_RUNS  directory for training and evaluation outputs
"""
import os
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

DATA = Path(os.environ.get("MAPNET_DATA", ROOT / "data"))
RUNS = Path(os.environ.get("MAPNET_RUNS", ROOT / "runs"))
RESULTS = ROOT / "results"
FIG = Path(os.environ.get("MAPNET_FIG", RUNS / "figures"))
