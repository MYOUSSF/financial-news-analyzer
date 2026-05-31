"""
Backtesting report generator — Markdown summary of backtest metrics.
"""
from datetime import datetime
from typing import Any, Dict


_REC_ORDER = ["STRONG BUY", "BUY", "HOLD", "SELL", "AVOID"]


def generate_backtest_report(metrics: Dict[str, Any]) -> str:
    """
    Render a Markdown summary table from ``BacktestEvaluator.compute_metrics()``
    output.

    Args:
        metrics: Dict returned by ``BacktestEvaluator.compute_metrics()``.

    Returns:
        Multi-line Markdown string.
    """
    total = metrics.get("total_evaluated", 0)
    overall_acc = metrics.get("overall_accuracy")
    acc_by_rec = metrics.get("accuracy_by_recommendation", {})
    avg_ret_by_rec = metrics.get("avg_return_by_recommendation", {})

    overall_acc_str = f"{overall_acc:.1%}" if overall_acc is not None else "N/A"

    lines = [
        "# Backtesting Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Total Evaluated:** {total}  ",
        f"**Overall Accuracy:** {overall_acc_str}",
        "",
        "## Accuracy by Recommendation",
        "",
        "| Recommendation | Accuracy | Avg Return (7d) |",
        "|:---------------|:--------:|----------------:|",
    ]

    # Fixed canonical order; any unexpected types appended alphabetically.
    all_recs = [r for r in _REC_ORDER if r in acc_by_rec]
    all_recs += sorted(r for r in acc_by_rec if r not in _REC_ORDER)

    for rec in all_recs:
        acc = acc_by_rec.get(rec)
        avg_ret = avg_ret_by_rec.get(rec)
        acc_str = f"{acc:.1%}" if acc is not None else "—"
        ret_str = f"{avg_ret:+.2%}" if avg_ret is not None else "—"
        lines.append(f"| {rec:<14} | {acc_str:>8} | {ret_str:>15} |")

    if total == 0:
        lines.extend(["", "_No evaluated recommendations yet._"])

    return "\n".join(lines) + "\n"
