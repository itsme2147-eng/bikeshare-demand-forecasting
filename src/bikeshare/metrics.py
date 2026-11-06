"""Forecast evaluation: point metrics, prediction intervals, planning trade-offs and the Diebold-Mariano test."""
from __future__ import annotations
import numpy as np
from scipy import stats


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def wape(y, p) -> float:
    """Weighted absolute percentage error: sum|e| / sum(y). Well defined when some hours have zero demand."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.abs(y - p).sum() / y.sum())


def bias_pct(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float((p - y).sum() / y.sum())


def r2(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def regression_report(y, p) -> dict:
    return {"mae": mae(y, p), "rmse": rmse(y, p), "wape": wape(y, p), "bias_pct": bias_pct(y, p), "r2": r2(y, p)}


def pinball_loss(y, q, tau: float) -> float:
    y, q = np.asarray(y, dtype=float), np.asarray(q, dtype=float)
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def interval_report(y, lo, hi) -> dict:
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi - lo))}


def planning_tradeoff(y, plan) -> dict:
    """If supply is planned at ``plan``: share of hours where demand exceeded it, and surplus as a share of demand."""
    y, plan = np.asarray(y, dtype=float), np.asarray(plan, dtype=float)
    return {"shortage_rate": float(np.mean(y > plan)), "surplus_share": float(np.maximum(plan - y, 0).sum() / y.sum()),
            "unmet_share": float(np.maximum(y - plan, 0).sum() / y.sum())}


def dm_test(y, p1, p2, lag: int = 23, loss: str = "abs") -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy (HAC variance, Harvey-Leybourne-Newbold small-sample fix).

    Returns (statistic, two-sided p-value). A **negative** statistic means forecast 1 has the lower loss.
    ``lag`` is the number of autocovariances used (h - 1 for an h-step forecast).
    """
    y, p1, p2 = (np.asarray(a, dtype=float) for a in (y, p1, p2))
    e1, e2 = y - p1, y - p2
    d = (np.abs(e1) - np.abs(e2)) if loss == "abs" else (e1 ** 2 - e2 ** 2)
    T = len(d)
    dc = d - d.mean()
    lrv = float(np.dot(dc, dc) / T)
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        lrv += 2 * w * float(np.dot(dc[k:], dc[:-k]) / T)
    if lrv <= 0 or not np.isfinite(lrv):
        return 0.0, 1.0
    h = lag + 1
    stat = d.mean() / np.sqrt(lrv / T)
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(p)
