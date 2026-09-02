import pandas as pd

from kaverentai.models.tune_renewal_rate import select_best_c


def test_select_best_c_uses_lowest_weighted_mae() -> None:
    results = pd.DataFrame(
        {
            "C": [0.1, 0.01, 1.0],
            "rolling_weighted_mae": [0.050, 0.045, 0.048],
        }
    )

    assert select_best_c(results) == 0.01
