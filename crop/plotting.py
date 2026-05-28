"""Minimal matplotlib plotting helpers used by experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def line_plot(df: pd.DataFrame, x: str, y: str, hue: str | None, output: str | Path, title: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if hue and hue in df.columns:
        for name, group in df.groupby(hue):
            group = group.sort_values(x)
            ax.plot(group[x], group[y], marker="o", label=str(name))
        ax.legend(frameon=False)
    else:
        group = df.sort_values(x)
        ax.plot(group[x], group[y], marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
