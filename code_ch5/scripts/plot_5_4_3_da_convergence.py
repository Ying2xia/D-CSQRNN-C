#!/usr/bin/env python3
"""Plot DA prediction convergence for all Section 5.4.2 settings."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT
from common.metrics import quantile_loss
from common.storage import ensure_dir


COLORS = {
    0.10: "#2563eb",
    0.25: "#059669",
    0.50: "#111827",
    0.75: "#ea580c",
    0.90: "#dc2626",
}
RAW_RE = re.compile(
    r"rep_(?P<rep>\d+)_s(?P<scenario>\d+)_(?P<error>normal|t3)_"
    r"(?P<censor_type>left|right|interval)_(?P<censor_rate>\d+)\.npz$"
)


@dataclass(frozen=True)
class Setting:
    scenario: int
    error: str
    censor_type: str
    censor_rate: float

    @property
    def tag(self) -> str:
        return f"s{self.scenario}_{self.error}_{self.censor_type}_{int(self.censor_rate * 100)}"

    @property
    def label(self) -> str:
        return (
            f"Scenario {self.scenario} | {self.error} | "
            f"{self.censor_type} censoring | {int(self.censor_rate * 100)}%"
        )


def raw_file_pattern(setting: Setting) -> str:
    return f"rep_*_s{setting.scenario}_{setting.error}_{setting.censor_type}_{int(setting.censor_rate * 100)}.npz"


def parse_raw_setting(path: Path) -> Setting | None:
    match = RAW_RE.match(path.name)
    if match is None:
        return None
    return Setting(
        scenario=int(match.group("scenario")),
        error=match.group("error"),
        censor_type=match.group("censor_type"),
        censor_rate=int(match.group("censor_rate")) / 100.0,
    )


def discover_settings(raw_dir: Path, args) -> list[Setting]:
    settings = sorted(
        {setting for file in raw_dir.glob("rep_*_s*.npz") if (setting := parse_raw_setting(file)) is not None},
        key=lambda s: (s.scenario, s.error, s.censor_type, s.censor_rate),
    )
    if args.scenario is not None:
        settings = [s for s in settings if s.scenario == args.scenario]
    if args.error is not None:
        settings = [s for s in settings if s.error == args.error]
    if args.censor_type is not None:
        settings = [s for s in settings if s.censor_type == args.censor_type]
    if args.censor_rate is not None:
        settings = [s for s in settings if float(s.censor_rate) == float(args.censor_rate)]
    if not settings:
        raise FileNotFoundError(f"No raw files found in {raw_dir} for the requested filters.")
    return settings


def collect_rows(raw_dir: Path, setting: Setting, args) -> pd.DataFrame:
    files = sorted(raw_dir.glob(raw_file_pattern(setting)))
    if args.rep:
        wanted = {int(r) for r in args.rep}
        files = [f for f in files if int(f.name.split("_")[1]) in wanted]
    if args.max_reps:
        files = files[: args.max_reps]
    if not files:
        raise FileNotFoundError(f"No raw files found in {raw_dir} for {setting.tag}.")

    rows: list[dict[str, float | int | str]] = []
    for file in files:
        rep = int(file.name.split("_")[1])
        z = np.load(file)
        y_test = z["y_test"]
        pred_iter = z["pred_da_iter"]
        pred_benchmark = z["pred_benchmark"]
        taus = [float(t) for t in z["target_taus"]]
        y_scale = float(np.std(y_test) + 1e-12)

        cumsum = np.cumsum(pred_iter, axis=0)
        for s_idx in range(pred_iter.shape[0]):
            s = s_idx + 1
            pred_cum = cumsum[s_idx] / s
            pred_prev = cumsum[s_idx - 1] / (s - 1) if s > 1 else None
            for j, tau in enumerate(taus):
                ql_bench = quantile_loss(y_test, pred_benchmark[:, j], tau)
                ql_iter = quantile_loss(y_test, pred_iter[s_idx, :, j], tau)
                ql_cum = quantile_loss(y_test, pred_cum[:, j], tau)
                if pred_prev is None:
                    rel_change = np.nan
                    abs_change = np.nan
                else:
                    diff = pred_cum[:, j] - pred_prev[:, j]
                    abs_change = float(np.sqrt(np.mean(diff**2)))
                    rel_change = abs_change / y_scale
                rows.append(
                    {
                        "rep": rep,
                        "scenario": setting.scenario,
                        "error": setting.error,
                        "censor_type": setting.censor_type,
                        "censor_rate": setting.censor_rate,
                        "S": s,
                        "tau": tau,
                        "ql_ratio_iter": ql_iter / ql_bench if ql_bench > 0 else np.nan,
                        "ql_ratio_cum": ql_cum / ql_bench if ql_bench > 0 else np.nan,
                        "pred_mean_iter": float(pred_iter[s_idx, :, j].mean()),
                        "pred_mean_cum": float(pred_cum[:, j].mean()),
                        "pred_change_rmse": abs_change,
                        "pred_change_relative": rel_change,
                    }
                )
    return pd.DataFrame(rows)


def summarize_trends(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "ql_ratio_iter",
        "ql_ratio_cum",
        "pred_mean_iter",
        "pred_mean_cum",
        "pred_change_rmse",
        "pred_change_relative",
    ]
    parts = []
    grouped = df.groupby(["scenario", "error", "censor_type", "censor_rate", "S", "tau"], dropna=False)
    for col in value_cols:
        tmp = grouped[col].agg(["mean", "std", "count"]).reset_index()
        tmp["se"] = tmp["std"] / np.sqrt(tmp["count"].clip(lower=1))
        tmp["metric"] = col
        tmp = tmp.rename(columns={"mean": "value_mean", "std": "value_sd", "count": "n"})
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _nice_bounds(values: np.ndarray, pad: float = 0.10, include: float | None = None) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if include is not None:
        values = np.append(values, include)
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        delta = abs(lo) * 0.1 + 1.0
        return lo - delta, hi + delta
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _format_tick(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _compressed_iteration_axis(
    values: np.ndarray,
    s_min: int,
    s_max: int,
    focus_iter: int,
    tail_width: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if s_max <= focus_iter:
        return values
    gap = max(2.0, tail_width * 0.25)
    tail_start = float(focus_iter) + gap
    tail_span = max(float(tail_width), 1.0)
    tail_den = max(float(s_max - focus_iter), 1.0)
    return np.where(
        values <= focus_iter,
        values,
        tail_start + (values - focus_iter) / tail_den * tail_span,
    )


def _iteration_ticks(
    s_min: int,
    s_max: int,
    focus_iter: int,
    tail_width: float,
) -> tuple[np.ndarray, list[str]]:
    if s_max <= focus_iter:
        x_tick_count = min(7, max(2, s_max - s_min + 1))
        ticks = np.linspace(s_min, s_max, num=x_tick_count)
        labels = [f"{int(round(v))}" for v in ticks]
        return ticks, labels

    focus_ticks = [s_min]
    focus_ticks.extend(v for v in range(10, focus_iter + 1, 10) if v >= s_min)
    if focus_iter not in focus_ticks:
        focus_ticks.append(focus_iter)
    if s_max not in focus_ticks:
        focus_ticks.append(s_max)
    ticks = _compressed_iteration_axis(np.array(focus_ticks, dtype=float), s_min, s_max, focus_iter, tail_width)
    labels = [f"{int(v)}" for v in focus_ticks]
    return ticks, labels


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Matplotlib is required for LaTeX-style math labels in convergence plots. "
            "Install matplotlib in the Python environment used to run this script."
        ) from exc
    return plt, FuncFormatter


def write_png_chart(
    summary: pd.DataFrame,
    metric: str,
    output: Path,
    setting: Setting,
    title: str,
    y_label: str,
    add_hline: float | None = None,
    focus_iter: int = 50,
    tail_width: float = 8.0,
) -> None:
    sub = summary[summary["metric"] == metric].copy()
    if sub.empty:
        return

    s_min, s_max = int(sub["S"].min()), int(sub["S"].max())
    y_values = sub["value_mean"].to_numpy(dtype=float)
    if "se" in sub.columns:
        y_values = np.concatenate(
            [
                y_values,
                (sub["value_mean"] - sub["se"].fillna(0.0)).to_numpy(dtype=float),
                (sub["value_mean"] + sub["se"].fillna(0.0)).to_numpy(dtype=float),
            ]
        )
    y_min, y_max = _nice_bounds(y_values, include=add_hline)

    plt, FuncFormatter = _load_matplotlib()
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(13.2, 8.2), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    fig.subplots_adjust(left=0.095, right=0.965, top=0.88, bottom=0.125)

    focus_iter = int(max(s_min, min(focus_iter, s_max)))
    tail_width = float(max(tail_width, 1.0))
    x_min = _compressed_iteration_axis(np.array([s_min], dtype=float), s_min, s_max, focus_iter, tail_width)[0]
    x_max = _compressed_iteration_axis(np.array([s_max], dtype=float), s_min, s_max, focus_iter, tail_width)[0]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    x_ticks, x_labels = _iteration_ticks(s_min, s_max, focus_iter, tail_width)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.linspace(y_min, y_max, num=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_tick(float(value))))
    ax.grid(axis="y", color="#e2e8f0", linewidth=1.0)
    ax.grid(axis="x", color="#f1f5f9", linewidth=0.9)
    ax.tick_params(axis="both", colors="#475569", labelsize=13, length=0, pad=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#64748b")
        ax.spines[spine].set_linewidth(1.4)

    ax.set_xlabel(r"DA iteration $s$", fontsize=16, fontweight="bold", color="#334155", labelpad=14)
    ax.set_ylabel(y_label, fontsize=18, color="#334155", labelpad=18)

    if add_hline is not None:
        ax.axhline(add_hline, color="#94a3b8", linewidth=1.8, linestyle=(0, (5, 5)), zorder=1)

    if s_max > focus_iter:
        gap = max(2.0, tail_width * 0.25)
        break_x = focus_iter + gap * 0.5
        trans = ax.get_xaxis_transform()
        ax.plot(
            [break_x - 0.55, break_x - 0.15],
            [-0.015, 0.035],
            transform=trans,
            color="#64748b",
            linewidth=1.2,
            clip_on=False,
        )
        ax.plot(
            [break_x + 0.15, break_x + 0.55],
            [-0.015, 0.035],
            transform=trans,
            color="#64748b",
            linewidth=1.2,
            clip_on=False,
        )

    for tau, group in sub.groupby("tau"):
        group = group.sort_values("S")
        group = group[np.isfinite(group["value_mean"])]
        if len(group) < 2:
            continue
        color = COLORS.get(round(float(tau), 2), "#64748b")
        s_values = group["S"].to_numpy(dtype=float)
        x = _compressed_iteration_axis(s_values, s_min, s_max, focus_iter, tail_width)
        mean = group["value_mean"].to_numpy(dtype=float)
        se = group.get("se", pd.Series(0.0, index=group.index)).fillna(0.0).to_numpy(dtype=float)
        if np.any(np.isfinite(se) & (se > 0)):
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.8,
            solid_capstyle="round",
            label=f"{float(tau):g}",
            zorder=3,
        )
        ax.scatter([x[-1]], [mean[-1]], s=34, color=color, edgecolor="#ffffff", linewidth=1.4, zorder=4)

    ax.legend(
        ncol=len(sorted(sub["tau"].unique())),
        loc="upper right",
        bbox_to_anchor=(0.965, 1.08),
        frameon=False,
        fontsize=13,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.6,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_setting_outputs(summary: pd.DataFrame, out_dir: Path, setting: Setting, args) -> list[str]:
    charts = [
        ("ql_ratio_cum", "Cumulative QL Ratio", r"$\mathrm{QL\_ratio}_{\tau}^{(s)}$", 1.0),
        ("pred_mean_cum", "Cumulative Mean Prediction", r"$\bar{\mu}_{\tau}^{(s)}$", None),
        ("pred_change_relative", "Relative Prediction Change", r"$\Delta_{\tau}^{(s)}$", None),
    ]
    paths = []
    for metric, title, y_label, hline in charts:
        path = out_dir / f"{setting.tag}_{metric}.png"
        write_png_chart(
            summary,
            metric,
            path,
            setting,
            title,
            y_label,
            add_hline=hline,
            focus_iter=args.focus_iter,
            tail_width=args.tail_width,
        )
        paths.append(str(path))
    return paths


def process_setting(raw_dir: Path, out_dir: Path, setting: Setting, args) -> dict[str, object]:
    raw_df = collect_rows(raw_dir, setting, args)
    summary = summarize_trends(raw_df)
    png_paths = write_setting_outputs(summary, out_dir, setting, args)
    return {
        "setting": setting.tag,
        "scenario": setting.scenario,
        "error": setting.error,
        "censor_type": setting.censor_type,
        "censor_rate": setting.censor_rate,
        "n_raw_rows": int(len(raw_df)),
        "n_replications": int(raw_df["rep"].nunique()),
        "png": png_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "results" / "5_4_2_ql_piw"))
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=None, help="Optional filter.")
    parser.add_argument("--error", choices=["normal", "t3"], default=None, help="Optional filter.")
    parser.add_argument("--censor-type", choices=["left", "right", "interval"], default=None, help="Optional filter.")
    parser.add_argument("--censor-rate", type=float, choices=[0.25, 0.50], default=None, help="Optional filter.")
    parser.add_argument("--rep", type=int, nargs="*", default=None, help="Optional specific replications.")
    parser.add_argument("--max-reps", type=int, default=None, help="Use the first N matching replications.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--focus-iter",
        type=int,
        default=50,
        help="Show DA iterations up to this value at full width; later iterations are compressed.",
    )
    parser.add_argument(
        "--tail-width",
        type=float,
        default=8.0,
        help="Display width used for iterations after --focus-iter.",
    )
    args = parser.parse_args()

    results = Path(args.results).expanduser().resolve()
    raw_dir = results / "raw"
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve() if args.out_dir else results / "convergence_plots")
    settings = discover_settings(raw_dir, args)

    for idx, setting in enumerate(settings, start=1):
        print(f"[{idx}/{len(settings)}] plotting {setting.tag}")
        process_setting(raw_dir, out_dir, setting, args)
    print(f"saved {len(settings) * 3} PNG chart(s) to {out_dir}")


if __name__ == "__main__":
    main()
