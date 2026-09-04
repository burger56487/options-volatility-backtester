import pandas as pd

from src.evaluation.model_selection import (
    calculate_selection_score,
    select_parameters,
)


def test_calculate_selection_score_penalises_small_samples():
    base = calculate_selection_score(
        sharpe=1.0,
        max_drawdown=0.0,
        turnover=0.0,
        trade_count=50,
    )
    small = calculate_selection_score(
        sharpe=1.0,
        max_drawdown=0.0,
        turnover=0.0,
        trade_count=5,
    )
    assert small < base


def test_selection_never_sees_test_rows():
    seen_indexes = []

    def evaluator(parameters, data):
        seen_indexes.append(set(data.index))
        return {
            "selection_score": float(parameters["score"]),
            "trade_count": len(data),
        }

    train = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
    validation = pd.DataFrame({"x": [4, 5]}, index=[3, 4])
    test = pd.DataFrame({"x": [6]}, index=[5])

    candidates = [{"score": 1.0}, {"score": 2.0}]
    selected, report = select_parameters(
        candidates=candidates,
        train_data=train,
        validation_data=validation,
        evaluator=evaluator,
        minimum_validation_trades=2,
    )

    seen = set().union(*seen_indexes)
    assert seen.isdisjoint(set(test.index))
    assert selected == {"score": 2.0}
    assert report["selected"].sum() == 1
