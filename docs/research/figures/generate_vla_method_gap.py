"""Generate the ArmBench VLA method-position and research-gap figure."""

from __future__ import annotations

import argparse
import pathlib
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


FIGURE_STEM = "armbench_vla_method_gap"
BACKGROUND = "#f4f6f7"
SURFACE = "#ffffff"
INK = "#152229"
MUTED = "#52616a"
LINE = "#cbd3d8"
HEADER = "#1b282f"
ARMBENCH = "#087f8c"
RTC = "#3b7f4f"
OFT = "#2f67a8"
DPPO = "#b06a18"
HIL = "#a94149"


METHODS = (
    {
        "name": "ArmBench",
        "status": "PROJECT EVIDENCE\nNOT PEER REVIEWED",
        "color": ARMBENCH,
        "adaptation": "Runtime + sampler ablation\nFrozen official pi0.5",
        "signal": "No training",
        "latency": "Measured-age suffix +\nhard-prefix overlap",
        "evidence": "LIBERO simulation\n4 suites / 40 tasks\npaired + manifest-bound",
        "formal": False,
    },
    {
        "name": "RTC",
        "status": "NEURIPS 2025\nFORMAL PAPER",
        "color": RTC,
        "adaptation": "Inference-time\naction-chunk inpainting",
        "signal": "No retraining",
        "latency": "Freeze committed actions;\ninpaint the continuation",
        "evidence": "Dynamic simulation +\nreal bimanual tasks",
        "formal": True,
    },
    {
        "name": "OpenVLA-OFT",
        "status": "RSS 2025\nFORMAL PAPER",
        "color": OFT,
        "adaptation": "VLA weight\nfine-tuning",
        "signal": "Offline imitation /\nL1 regression",
        "latency": "Faster parallel decoding;\nasync staleness not central",
        "evidence": "LIBERO simulation +\nreal bimanual ALOHA",
        "formal": True,
    },
    {
        "name": "DPPO",
        "status": "ICLR 2025\nFORMAL PAPER",
        "color": DPPO,
        "adaptation": "Diffusion-policy\nRL fine-tuning",
        "signal": "Policy gradients /\nreward",
        "latency": "Not an asynchronous\nruntime method",
        "evidence": "Simulation + zero-shot\nhardware deployment",
        "formal": True,
    },
    {
        "name": "HIL-SERL",
        "status": "SCIENCE ROBOTICS 2025\nFORMAL PAPER",
        "color": HIL,
        "adaptation": "Real-world policy\nlearning",
        "signal": "Demos + human\ncorrections + RL",
        "latency": "Not an action-chunk\nlatency method",
        "evidence": "Real robot +\ndexterous tasks",
        "formal": True,
    },
)


def _text(
    ax,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 9.0,
    color: str = INK,
    weight: str = "normal",
    ha: str = "left",
    va: str = "center",
    linespacing: float = 1.25,
    **kwargs,
) -> None:
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=linespacing,
        **kwargs,
    )


def _rounded_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = SURFACE,
    edgecolor: str = LINE,
    linewidth: float = 1.0,
    linestyle: str = "solid",
    radius: float = 0.008,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    return patch


def _draw_method_table(ax) -> None:
    x0 = 0.035
    x1 = 0.665
    top = 0.842
    header_h = 0.064
    row_h = 0.124
    columns = (x0, 0.192, 0.326, 0.425, 0.548, x1)
    headers = (
        "METHOD / STATUS",
        "ADAPTATION LOCUS",
        "LEARNING SIGNAL",
        "LATENCY QUESTION",
        "EVIDENCE DOMAIN",
    )

    ax.add_patch(Rectangle((x0, top), x1 - x0, header_h, color=HEADER))
    for index, label in enumerate(headers):
        _text(
            ax,
            columns[index] + 0.009,
            top + header_h / 2,
            label,
            size=7.5,
            color="#f7fafb",
            weight="bold",
        )

    for row_index, method in enumerate(METHODS):
        y = top - (row_index + 1) * row_h
        linestyle = "solid" if method["formal"] else (0, (4, 3))
        _rounded_box(
            ax,
            x0,
            y + 0.006,
            x1 - x0,
            row_h - 0.012,
            edgecolor=method["color"],
            linewidth=1.8 if row_index == 0 else 1.0,
            linestyle=linestyle,
            radius=0.006,
        )
        ax.add_patch(
            Rectangle(
                (x0, y + 0.006),
                0.006,
                row_h - 0.012,
                facecolor=method["color"],
                edgecolor="none",
            )
        )
        for boundary in columns[1:-1]:
            ax.plot(
                (boundary, boundary),
                (y + 0.018, y + row_h - 0.018),
                color=LINE,
                linewidth=0.7,
            )

        _text(
            ax,
            columns[0] + 0.012,
            y + row_h * 0.68,
            method["name"],
            size=11.0,
            color=method["color"],
            weight="bold",
        )
        _text(
            ax,
            columns[0] + 0.012,
            y + row_h * 0.35,
            method["status"],
            size=6.8,
            color=MUTED,
            weight="bold",
        )
        for column_index, key in enumerate(
            ("adaptation", "signal", "latency", "evidence"), start=1
        ):
            _text(
                ax,
                columns[column_index] + 0.009,
                y + row_h / 2,
                method[key],
                size=7.7,
                color=INK,
            )

    _text(
        ax,
        x0,
        0.135,
        "Solid border: formal venue/journal paper    Dashed border: project evidence, not a publication",
        size=7.4,
        color=MUTED,
    )


def _bullet_block(ax, x: float, y: float, lines: tuple[str, ...], color: str) -> None:
    for index, line in enumerate(lines):
        line_y = y - index * 0.039
        ax.add_patch(
            Rectangle(
                (x, line_y - 0.005),
                0.008,
                0.008,
                facecolor=color,
                edgecolor="none",
            )
        )
        _text(ax, x + 0.015, line_y, line, size=7.8, color=INK)


def _draw_gap_panel(ax) -> None:
    x = 0.695
    width = 0.27
    _rounded_box(ax, x, 0.558, width, 0.348, edgecolor=ARMBENCH, linewidth=1.4)
    _text(ax, x + 0.018, 0.875, "ARMBENCH TODAY", size=8.0, color=ARMBENCH, weight="bold")
    _text(
        ax,
        x + 0.018,
        0.838,
        "Strong systems evidence,\nVJP guidance pilot complete",
        size=12.0,
        weight="bold",
        va="top",
    )
    _bullet_block(
        ax,
        x + 0.020,
        0.770,
        (
            "Frozen official pi0.5 checkpoint",
            "Policy-internal hard + VJP conditioning",
            "Frozen protocols + matched conditions",
            "Exact transition transcript + videos",
            "Source/checkpoint/video attestation",
        ),
        ARMBENCH,
    )

    _rounded_box(ax, x, 0.205, width, 0.318, edgecolor="#8b3d44", linewidth=1.4)
    _text(ax, x + 0.018, 0.493, "TOP-CONFERENCE GAP", size=8.0, color="#8b3d44", weight="bold")
    _bullet_block(
        ax,
        x + 0.020,
        0.449,
        (
            "Independent inference / control clocks",
            "Powered held-out RTC efficacy test",
            "More than one VLA checkpoint",
            "Real-robot closed-loop validation",
            "Task efficacy beyond a small pilot",
        ),
        "#8b3d44",
    )
    _text(
        ax,
        x + 0.020,
        0.246,
        "These gaps test generality and deployment realism;\nthey do not imply that RL is mandatory.",
        size=7.5,
        color=MUTED,
        va="top",
    )

    _rounded_box(
        ax,
        x,
        0.070,
        width,
        0.100,
        facecolor="#eaf3ed",
        edgecolor=RTC,
        linewidth=1.2,
    )
    _text(ax, x + 0.018, 0.145, "NEAREST SCIENTIFIC COMPARATOR", size=7.6, color=RTC, weight="bold")
    _text(
        ax,
        x + 0.018,
        0.112,
        "RTC: training-free, asynchronous action-chunk\nexecution with simulation and real-robot evidence.",
        size=7.7,
        color=INK,
    )


def build_figure(output_directory: pathlib.Path, dpi: int = 240) -> list[pathlib.Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "svg.hashsalt": "armbench-vla-method-gap-v1",
        }
    )
    fig = plt.figure(figsize=(15.2, 8.6), facecolor=BACKGROUND)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _text(
        ax,
        0.035,
        0.965,
        "ArmBench in the robot-policy adaptation landscape",
        size=22.0,
        color=INK,
        weight="bold",
        va="top",
    )
    _text(
        ax,
        0.035,
        0.921,
        "Runtime reliability and statistical evidence are distinct contributions from supervised fine-tuning, RL adaptation, and human-in-the-loop learning.",
        size=9.8,
        color=MUTED,
        va="top",
    )

    _draw_method_table(ax)
    _draw_gap_panel(ax)

    footer = (
        "Status checked 2026-08-05. [1] OpenVLA-OFT: RSS 2025, DOI 10.15607/RSS.2025.XXI.017.  "
        "[2] DPPO: ICLR 2025, arXiv:2409.00588.  [3] HIL-SERL: Science Robotics 2025, DOI 10.1126/scirobotics.ads5033.  "
        "[4] RTC: NeurIPS 2025, arXiv:2506.07339. ArmBench scope follows the repository's frozen evidence; see the adjacent source note."
    )
    _text(
        ax,
        0.035,
        0.017,
        textwrap.fill(footer, width=210),
        size=6.6,
        color=MUTED,
        va="bottom",
        linespacing=1.2,
    )

    outputs = [
        output_directory / f"{FIGURE_STEM}.png",
        output_directory / f"{FIGURE_STEM}.pdf",
        output_directory / f"{FIGURE_STEM}.svg",
    ]
    fig.savefig(
        outputs[0],
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Software": "ArmBench matplotlib generator"},
    )
    fig.savefig(
        outputs[1],
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={
            "Title": "ArmBench VLA method gap",
            "Author": "ArmBench",
            "Creator": "ArmBench matplotlib generator",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        outputs[2],
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Title": "ArmBench VLA method gap", "Date": None},
    )
    plt.close(fig)
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--output-directory",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent,
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.dpi < 72:
        raise ValueError("dpi must be at least 72")
    for output in build_figure(args.output_directory.resolve(), args.dpi):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
