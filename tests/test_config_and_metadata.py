import json
from pathlib import Path

import pytest

from src.config import load_config, validate_config
from src.research_guardrails import validate_research_claims
from src.run_context import initialise_run
from src.run_metadata import (
    create_config_hash,
    create_run_metadata,
    save_run_metadata,
)


def test_default_config_can_be_loaded():
    config = load_config("configs/default.yaml")
    assert config["data"]["underlying"]["data_type"] == "real"
    assert config["data"]["options"]["data_type"] == "synthetic"
    assert config["execution"]["mode"] == "simulated"


def test_invalid_option_data_type_is_rejected():
    config = load_config("configs/default.yaml")
    config["data"]["options"]["data_type"] = "unknown"
    with pytest.raises(ValueError):
        validate_config(config)


def test_config_hash_is_stable():
    config_a = {"b": 2, "a": 1}
    config_b = {"a": 1, "b": 2}
    assert create_config_hash(config_a) == create_config_hash(config_b)


def test_metadata_contains_research_boundary():
    config = load_config("configs/default.yaml")
    metadata = create_run_metadata(config)
    assert metadata["data"]["underlying_data_type"] == "real"
    assert metadata["data"]["option_data_type"] == "synthetic"
    assert metadata["execution"]["mode"] == "simulated"
    assert metadata["research"]["evaluation_mode"] == "in_sample"
    assert metadata["reproducibility"]["config_hash"]


def test_metadata_can_be_saved(tmp_path: Path):
    config = load_config("configs/default.yaml")
    metadata = create_run_metadata(config)
    path = save_run_metadata(
        metadata=metadata,
        output_directory=tmp_path,
    )
    assert path.exists()
    with path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["run_id"] == metadata["run_id"]


def test_synthetic_data_cannot_be_called_real_backtest():
    config = load_config("configs/default.yaml")
    with pytest.raises(ValueError):
        validate_research_claims(
            config=config,
            run_label="真实期权市场回测",
        )


def test_synthetic_research_label_is_allowed():
    config = load_config("configs/default.yaml")
    validate_research_claims(
        config=config,
        run_label="合成期权波动率策略研究",
    )


def test_initialise_run_creates_snapshots(tmp_path: Path):
    config = load_config("configs/default.yaml")
    config["output"]["directory"] = str(tmp_path)
    context = initialise_run(
        config=config,
        config_path="configs/default.yaml",
    )
    assert context.output_directory.exists()
    assert (context.output_directory / "run_metadata.json").exists()
    assert (context.output_directory / "config_snapshot.yaml").exists()
    assert (context.output_directory / "research_boundary.json").exists()
