import numpy as np
import pytest

from bikeshare import dm_test, interval_report, planning_tradeoff, regression_report, wape
from bikeshare.metrics import pinball_loss


def test_wape_and_report_known_values():
    y, p = np.array([10, 20, 0, 30.0]), np.array([12, 18, 2, 30.0])
    assert wape(y, p) == pytest.approx(6 / 60)
    r = regression_report(y, p)
    assert r["mae"] == pytest.approx(1.5) and r["bias_pct"] == pytest.approx(2 / 60)
    assert r["rmse"] == pytest.approx(np.sqrt((4 + 4 + 4) / 4))


def test_wape_is_defined_when_demand_contains_zeros():
    assert np.isfinite(wape(np.array([0, 0, 5.0]), np.array([1, 0, 4.0])))


def test_interval_coverage_and_width():
    r = interval_report([1, 2, 3, 4], [0, 0, 4, 3], [2, 1, 5, 5])
    assert r["coverage"] == pytest.approx(0.5) and r["mean_width"] == pytest.approx((2 + 1 + 1 + 2) / 4)


def test_planning_tradeoff_directions():
    y = np.array([10, 20, 30, 40.0])
    low, high = planning_tradeoff(y, y * 0.8), planning_tradeoff(y, y * 1.3)
    assert low["shortage_rate"] == 1.0 and high["shortage_rate"] == 0.0
    assert high["surplus_share"] == pytest.approx(0.3) and low["unmet_share"] == pytest.approx(0.2)


def test_pinball_loss_is_asymmetric():
    assert pinball_loss([10.0], [5.0], 0.9) == pytest.approx(4.5)
    assert pinball_loss([10.0], [15.0], 0.9) == pytest.approx(0.5)


def test_dm_test_detects_clearly_better_forecast_and_sign():
    rng = np.random.default_rng(0); y = rng.normal(100, 10, 800)
    good, bad = y + rng.normal(0, 2, 800), y + rng.normal(0, 8, 800)
    stat, p = dm_test(y, good, bad, lag=23)
    assert stat < 0 and p < 0.001                              # forecast 1 (good) has lower loss
    stat2, _ = dm_test(y, bad, good, lag=23)
    assert stat2 > 0


def test_dm_test_does_not_reject_for_equally_good_forecasts():
    rng = np.random.default_rng(1); y = rng.normal(100, 10, 800)
    a, b = y + rng.normal(0, 5, 800), y + rng.normal(0, 5, 800)
    assert dm_test(y, a, b, lag=23)[1] > 0.01
    assert dm_test(y, a, a)[1] == 1.0                           # identical forecasts: no difference
