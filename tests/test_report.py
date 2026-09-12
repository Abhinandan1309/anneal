"""Report rendering: compact labels, Markdown, and the dependency-free SVG."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from anneal.core.artifact import TransformRecord
from anneal.core.ledger import Ledger
from anneal.report import (
    compact_label,
    frontier_svg,
    markdown_report,
    wilson_halfwidth_pp,
    write_report,
)
from conftest import make_trial


def test_baseline_label_says_fp32():
    assert compact_label(make_trial(0)) == "baseline (fp32)"


def test_compact_label_abbreviates_a_long_recipe():
    trial = make_trial(
        1,
        lineage=(
            TransformRecord(
                "quantize_static_int8",
                {
                    "per_channel": True,
                    "reduce_range": False,
                    "calibrate_method": "minmax",
                    "calib_samples": 64,
                },
            ),
        ),
    )
    label = compact_label(trial)
    assert label == "static-int8[pc,minmax]"
    assert len(label) < len(trial.artifact.label)


def test_compact_label_distinguishes_per_channel_from_per_tensor():
    def label(per_channel: bool) -> str:
        return compact_label(
            make_trial(
                1, lineage=(TransformRecord("quantize_dynamic_int8", {"per_channel": per_channel}),)
            )
        )

    assert label(True) == "dyn-int8[pc]"
    assert label(False) == "dyn-int8[pt]"


def test_compact_label_shows_chained_transforms():
    trial = make_trial(
        1,
        lineage=(
            TransformRecord("graph_optimize", {"level": "all"}),
            TransformRecord("quantize_dynamic_sensitive", {"skip_top_k": 2, "skip_first_last": True}),
        ),
    )
    assert compact_label(trial) == "fuse[all] > sel-int8[k=2,+stem/head]"


# ----- markdown ------------------------------------------------------------


def test_wilson_interval_shrinks_as_the_eval_set_grows():
    small = wilson_halfwidth_pp(0.668, 256)
    large = wilson_halfwidth_pp(0.668, 25_600)
    assert small is not None and large is not None
    assert small > large
    # 256 images cannot resolve a 2pp difference; the README claims this, so pin it.
    assert small > 2.0


def test_wilson_interval_stays_finite_at_the_extremes():
    assert wilson_halfwidth_pp(1.0, 100) > 0
    assert wilson_halfwidth_pp(0.0, 100) > 0


def test_wilson_interval_is_undefined_without_data():
    assert wilson_halfwidth_pp(None, 256) is None
    assert wilson_halfwidth_pp(0.5, 0) is None


def test_markdown_states_the_accuracy_resolution(ledger: Ledger):
    md = markdown_report(ledger)
    assert "Accuracy resolution:" in md
    assert "tied, not ranked" in md


def test_markdown_report_has_the_sections_a_reader_needs(ledger: Ledger):
    md = markdown_report(ledger)
    for section in ("## Setup", "## All trials", "## Pareto frontier", "## Reproducing"):
        assert section in md
    assert ledger.run_id in md


def test_markdown_reports_the_measurement_protocol(ledger: Ledger):
    ledger.config.update({"warmup": 10, "runs": 50, "batch_size": 1})
    md = markdown_report(ledger)
    assert "10 warm-up discarded, 50 timed runs" in md


def test_markdown_flags_synthetic_accuracy_as_meaningless(ledger: Ledger):
    ledger.config["evalset_synthetic"] = True
    assert "meaningless by construction" in markdown_report(ledger)


def test_markdown_surfaces_failures_rather_than_hiding_them(ledger: Ledger, q8):
    ledger.add(make_trial(4, lineage=(q8,), error="ConvInteger not implemented"))
    md = markdown_report(ledger)
    assert "## Failed trials" in md
    assert "ConvInteger not implemented" in md


def test_markdown_surfaces_portability_warnings(ledger: Ledger):
    ledger.trials[1].artifact.meta["portability_warning"] = "NCHWc layout is CPU-specific"
    md = markdown_report(ledger)
    assert "## Portability caveats" in md
    assert "NCHWc layout is CPU-specific" in md


def test_markdown_omits_the_caveats_section_when_there_is_nothing_to_warn_about(ledger: Ledger):
    assert "## Portability caveats" not in markdown_report(ledger)


def test_markdown_uses_full_recipes_not_abbreviations(ledger: Ledger):
    # The report is the archival record; it must stay unambiguous.
    assert "quantize_dynamic_int8(per_channel=True)" in markdown_report(ledger)


# ----- svg -----------------------------------------------------------------


def test_frontier_svg_is_well_formed_xml(ledger: Ledger):
    root = ElementTree.fromstring(frontier_svg(ledger))
    assert root.tag.endswith("svg")


def test_frontier_svg_plots_every_scored_trial(ledger: Ledger):
    root = ElementTree.fromstring(frontier_svg(ledger))
    circles = root.findall(".//{http://www.w3.org/2000/svg}circle")
    assert len(circles) == len(ledger.scored())


def test_frontier_svg_keeps_every_point_inside_the_canvas(ledger: Ledger):
    # Axis padding is computed from the data, so a bad range calculation silently
    # scatters points off the edge of the image rather than raising.
    width, height = 760, 440
    root = ElementTree.fromstring(frontier_svg(ledger, width=width, height=height))
    for circle in root.findall(".//{http://www.w3.org/2000/svg}circle"):
        cx, cy, r = (float(circle.get(k)) for k in ("cx", "cy", "r"))
        assert 0 <= cx - r and cx + r <= width
        assert 0 <= cy - r and cy + r <= height


def test_frontier_svg_marks_the_baseline_distinctly(ledger: Ledger):
    root = ElementTree.fromstring(frontier_svg(ledger))
    fills = [c.get("fill") for c in root.findall(".//{http://www.w3.org/2000/svg}circle")]
    assert fills.count("#ffffff") == 1, "exactly one hollow marker, for the baseline"


def test_frontier_svg_degrades_gracefully_with_no_data():
    svg = frontier_svg(Ledger())
    assert "no scored trials" in svg
    ElementTree.fromstring(svg)


def test_frontier_svg_survives_a_single_point():
    led = Ledger()
    led.add(make_trial(0, latency=40.0, accuracy=0.9))
    ElementTree.fromstring(frontier_svg(led))


def test_frontier_svg_survives_identical_points():
    # Degenerate axis ranges must not divide by zero.
    led = Ledger()
    led.add(make_trial(0, latency=40.0, accuracy=0.9))
    led.add(make_trial(1, lineage=(TransformRecord("x"),), latency=40.0, accuracy=0.9))
    ElementTree.fromstring(frontier_svg(led))


# ----- writing -------------------------------------------------------------


def test_write_report_emits_all_three_artifacts(ledger: Ledger, tmp_path: Path):
    paths = write_report(ledger, tmp_path / "out")
    assert set(paths) == {"ledger", "report", "svg"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    assert "frontier.svg" in paths["report"].read_text(encoding="utf-8")


def test_written_ledger_can_be_loaded_back(ledger: Ledger, tmp_path: Path):
    paths = write_report(ledger, tmp_path / "out")
    restored = Ledger.load(paths["ledger"])
    assert len(restored.trials) == len(ledger.trials)
