"""Run SIREN-PINN inversion on Profile 1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_inversion import run

if __name__ == "__main__":
    run("configs/config_profile1.yaml")