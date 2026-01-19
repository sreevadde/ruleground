"""Tests for config system."""

import pytest
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from ruleground.utils.config import (
    load_config,
    load_config_with_defaults,
    config_to_dict,
    DEFAULT_CONFIG,
)


class TestConfig:
    def test_default_config_structure(self):
        assert "model" in DEFAULT_CONFIG
        assert "training" in DEFAULT_CONFIG
        assert "grpo" in DEFAULT_CONFIG
        assert "rsa" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG.model.rgm.num_heads == 8
        assert DEFAULT_CONFIG.training.gamma == 0.5

    def test_load_yaml(self, tmp_path):
        yaml_content = """
model:
  encoder:
    name: test-model
    freeze: true
  rgm:
    num_heads: 4
"""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml_content)
        config = load_config(config_path)
        assert config.model.encoder.name == "test-model"
        assert config.model.rgm.num_heads == 4

    def test_load_with_overrides(self, tmp_path):
        yaml_content = """
training:
  lr: 0.001
  batch_size: 16
"""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml_content)
        config = load_config(config_path, overrides=["training.lr=0.0001"])
        assert config.training.lr == 0.0001
        assert config.training.batch_size == 16

    def test_load_with_defaults(self, tmp_path):
        defaults = """
model:
  encoder:
    name: default-model
    freeze: true
training:
  lr: 0.001
"""
        override = """
training:
  lr: 0.0001
"""
        defaults_path = tmp_path / "defaults.yaml"
        override_path = tmp_path / "override.yaml"
        defaults_path.write_text(defaults)
        override_path.write_text(override)

        config = load_config_with_defaults(override_path, defaults_path)
        assert config.model.encoder.name == "default-model"  # from defaults
        assert config.training.lr == 0.0001  # overridden

    def test_config_to_dict(self):
        d = config_to_dict(DEFAULT_CONFIG)
        assert isinstance(d, dict)
        assert isinstance(d["model"], dict)
        assert d["model"]["rgm"]["num_heads"] == 8
