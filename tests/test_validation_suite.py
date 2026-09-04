"""The model-validation suite must stay green as one checkable unit."""

import pytest

from src.validation.suite import run_validation_suite

pytestmark = pytest.mark.model_validation


@pytest.fixture(scope="module")
def validation_report() -> dict:
    return run_validation_suite()


def test_validation_suite_all_checks_pass(validation_report: dict) -> None:
    failures = [
        check["name"]
        for check in validation_report["checks"]
        if check["status"] != "pass"
    ]
    assert failures == [], failures


def test_validation_suite_covers_core_models(
    validation_report: dict,
) -> None:
    names = {
        check["name"]
        for check in validation_report["checks"]
    }
    assert {
        "crr_convergence_to_bs",
        "crank_nicolson_to_bs",
        "monte_carlo_se_scaling",
        "heston_degenerates_to_bs",
        "merton_degenerates_to_bs",
        "svi_real_chain_calibration_quality",
        "var_backtest_on_spy",
    } <= names
