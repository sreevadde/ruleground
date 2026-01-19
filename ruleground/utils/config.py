"""
Configuration System

OmegaConf-based hierarchical configuration for RuleGround.
Supports YAML files with merge, override, and interpolation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(
    path: Union[str, Path],
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load a YAML config with optional CLI overrides.

    Args:
        path:      Path to YAML config file.
        overrides: List of dotpath overrides, e.g. ['training.lr=3e-4'].

    Returns:
        Merged DictConfig.
    """
    base = OmegaConf.load(str(path))

    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, override_conf)

    OmegaConf.resolve(base)
    return base


def load_config_with_defaults(
    path: Union[str, Path],
    defaults_path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load config merged on top of a defaults file.

    Args:
        path:          Primary config YAML.
        defaults_path: Base defaults YAML (merged first).
        overrides:     CLI dotlist overrides.

    Returns:
        Merged DictConfig.
    """
    configs = []

    if defaults_path:
        configs.append(OmegaConf.load(str(defaults_path)))

    configs.append(OmegaConf.load(str(path)))

    if overrides:
        configs.append(OmegaConf.from_dotlist(overrides))

    merged = OmegaConf.merge(*configs)
    OmegaConf.resolve(merged)
    return merged


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert DictConfig to plain dict (for logging/serialization)."""
    return OmegaConf.to_container(config, resolve=True)


# Default configuration for quick experimentation
DEFAULT_CONFIG = OmegaConf.create({
    "model": {
        "encoder": {
            "name": "MCG-NJU/videomae-base",
            "freeze": True,
        },
        "rgm": {
            "num_heads": 8,
            "hidden_dim": 256,
            "dropout": 0.1,
            "use_rope": True,
            "use_flash_attn": True,
        },
        "reasoning_head": {
            "hidden_dim": 256,
            "dropout": 0.1,
        },
        "num_q1_classes": 2,
        "num_q2_classes": 17,
    },
    "data": {
        "root": "data/sportr",
        "num_frames": 16,
        "frame_size": [224, 224],
        "num_workers": 4,
    },
    "training": {
        "batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "max_epochs": 30,
        "gamma": 0.5,
        "delta": 0.1,
        "pretrain_epochs": 10,
        "use_grpo": True,
        "grpo_epochs": 5,
        "use_rsa": True,
        "rsa_epochs": 5,
    },
    "grpo": {
        "group_size": 8,
        "clip_ratio": 0.2,
        "kl_coef": 0.1,
        "predicate_dropout": 0.1,
    },
    "rsa": {
        "alpha": 0.1,
        "lambda_risk": 0.3,
        "fp_penalty": 2.0,
    },
    "logging": {
        "project": "ruleground",
        "use_wandb": False,
        "log_every_n_steps": 50,
    },
})
