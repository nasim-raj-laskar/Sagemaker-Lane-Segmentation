"""src/utils/config.py — Hydra/OmegaConf config loading outside of @hydra.main."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def load_config(
    config_dir: str | Path | None = None,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load a Hydra config from config_dir without launching a Hydra app.

    Useful for scripts, notebooks, and tests that need the resolved config
    without going through @hydra.main.

    Args:
        config_dir: Absolute path to the configs/ directory.
                    Defaults to <project_root>/configs.
        config_name: Root config file name (without .yaml).
        overrides:   List of Hydra override strings, e.g. ["training.epochs=5"].

    Returns:
        Fully resolved DictConfig.
    """
    if config_dir is None:
        config_dir = Path(__file__).parents[2] / "configs"
    config_dir = str(Path(config_dir).resolve())

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg


def print_config(cfg: DictConfig) -> None:
    """Pretty-print the resolved config to stdout."""
    print(OmegaConf.to_yaml(cfg))


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Persist the resolved config as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
