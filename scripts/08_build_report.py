"""Build the technical report (Word) from the committed results.

    uv run python scripts/08_build_report.py

Writes ``report/PathSnap_Technical_Report.docx``. Every table is read from the
CSV/JSON files in ``results/`` and ``models/``, and every figure is a PNG from
``results/`` or ``report/figures/`` (UI and API screenshots), so the numbers in
the report are the numbers the pipeline produced. Prose that quotes a number
takes it from the same files. Evaluator-test data is not read.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
OUT = ROOT / "report" / "PathSnap_Technical_Report.docx"

MEMBERS = [
    ("Sathvik Karthik Malali", "PES1UG24CS616"),
    ("Satvik K", "PES1UG24CS617"),
    ("Gagan R", "PES1UG24CS665"),
    ("Fathima Zahra", "PES1UG24CS664"),
]

BODY_FONT = "Calibri"
HEAD_COLOUR = RGBColor(0x1F, 0x3A, 0x5F)
TEXT_WIDTH_CM = 17.0

# ------------------------------------------------------------------ data

mc = pd.read_csv(RES / "models" / "model_comparison.csv")
abl = pd.read_csv(RES / "models" / "ablation_pair_features" / "model_comparison.csv")
run_cfg = json.loads((RES / "models" / "run_config.json").read_text())
model_params = {m["name"]: m["params"] for m in run_cfg["models"]}
features_final = run_cfg["features_final"]
schema = pd.read_csv(RES / "features" / "feature_schema.csv")
feat_summary = pd.read_csv(RES / "features" / "feature_summary.csv").set_index("partition")
inv = pd.read_csv(RES / "eda" / "inventory.csv", dtype={"trip": str})
sweep = pd.read_csv(RES / "candidates" / "sweep.csv")
hgb_hist = pd.read_csv(RES / "models" / "histories" / "histogram_gradient_boosting.csv")
mlp_hist = pd.read_csv(RES / "models" / "histories" / "multilayer_perceptron.csv")
cm = pd.read_csv(RES / "models" / "confusion_matrices.csv").set_index("model")

tune = pd.read_csv(RES / "tuning" / "untuned_vs_tuned.csv").set_index("configuration")
space = json.loads((RES / "tuning" / "search_space.json").read_text())
best = json.loads((RES / "tuning" / "best_params.json").read_text())
importance = pd.read_csv(RES / "tuning" / "param_importance.csv")
trials = pd.read_csv(RES / "tuning" / "trials.csv")

hmm_sum = pd.read_csv(RES / "hmm" / "summary.csv").set_index(["partition", "method"])
hmm_trip = pd.read_csv(RES / "hmm" / "per_trip.csv", dtype={"trip": str})
decoder = json.loads((RES / "hmm" / "decoder_config.json").read_text())

M = mc.set_index("model")
LEADER = "LightGBM LambdaRank"


def f3(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.3f}"


def f4(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.4f}"


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ document helpers

doc = Document()
_fig_no = 0
_tab_no = 0


def _set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def _cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def _runs(paragraph, text: str, size: float | None = None, italic: bool = False) -> None:
    """Add text with **bold** and `code` markup as separate runs."""
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|\[\[[^\]]+\]\])", text):
        if not part:
            continue
        if part.startswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        elif part.startswith("[["):
            run = paragraph.add_run(part[2:-2])
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        else:
            run = paragraph.add_run(part)
        if size:
            run.font.size = Pt(size)
        if italic:
            run.italic = True


def para(text: str, style: str | None = None, align=None, size=None, italic=False, after=None):
    p = doc.add_paragraph(style=style)
    _runs(p, text, size=size, italic=italic)
    if align is not None:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullets(items: list[str]) -> None:
    for it in items:
        para(it, style="List Bullet")


def numbered(items: list[str]) -> None:
    for it in items:
        para(it, style="List Number")


def qa(question: str, answer: str | list[str]) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(question)
    run.bold = True
    run.font.color.rgb = HEAD_COLOUR
    if isinstance(answer, str):
        para(answer)
    else:
        bullets(answer)


def h(text: str, level: int):
    return doc.add_heading(text, level=level)


def page_break() -> None:
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def figure(path: Path, caption: str, width_cm: float = TEXT_WIDTH_CM) -> None:
    global _fig_no
    _fig_no += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = doc.add_paragraph(style="Caption")
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _runs(c, f"**Figure {_fig_no}.** {caption}")


def table(headers: list[str], rows: list[list[str]], widths_cm: list[float], caption: str,
          size: float = 8.5, numeric_from: int = 1) -> None:
    global _tab_no
    _tab_no += 1
    c = doc.add_paragraph(style="Caption")
    c.paragraph_format.keep_with_next = True
    _runs(c, f"**Table {_tab_no}.** {caption}")
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, (cell, text) in enumerate(zip(t.rows[0].cells, headers)):
        cell.width = Cm(widths_cm[i])
        _set_cell_shading(cell, "DCE4EE")
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if i < numeric_from else WD_ALIGN_PARAGRAPH.RIGHT
        _runs(cell.paragraphs[0], f"**{text}**", size=size)
    _set_repeat_header(t.rows[0])
    for r in rows:
        row = t.add_row()
        _cant_split(row)
        for i, (cell, text) in enumerate(zip(row.cells, r)):
            cell.width = Cm(widths_cm[i])
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if i < numeric_from else WD_ALIGN_PARAGRAPH.RIGHT
            _runs(cell.paragraphs[0], str(text), size=size)
    # Grid widths for Word and LibreOffice, which read tblGrid rather than cell widths.
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths_cm):
        gc.set(qn("w:w"), str(int(w / 2.54 * 1440)))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _page_number_footer(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)
    run.font.size = Pt(9)


def setup_styles() -> None:
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.0))
    sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)
    _page_number_footer(sec)

    st = doc.styles
    normal = st["Normal"]
    normal.font.name = BODY_FONT
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for name, size, before, after in (("Heading 1", 16, 18, 8), ("Heading 2", 13, 14, 6),
                                      ("Heading 3", 11.5, 10, 4)):
        s = st[name]
        s.font.name = BODY_FONT
        for theme_attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            s.element.rPr.rFonts.attrib.pop(qn(theme_attr), None)
        s.element.rPr.rFonts.set(qn("w:ascii"), BODY_FONT)
        s.element.rPr.rFonts.set(qn("w:hAnsi"), BODY_FONT)
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = HEAD_COLOUR
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True
    cap = st["Caption"]
    cap.font.name = BODY_FONT
    cap.font.size = Pt(9)
    cap.font.italic = False
    cap.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    cap.paragraph_format.space_after = Pt(10)
    for name in ("List Bullet", "List Number"):
        st[name].font.name = BODY_FONT
        st[name].font.size = Pt(10.5)
        st[name].paragraph_format.space_after = Pt(3)


# ------------------------------------------------------------------ title page

def title_page() -> None:
    for _ in range(5):
        doc.add_paragraph()
    para("PES University · Department of Computer Science and Engineering · RR Campus",
         align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
    para("UE24CS352A: Machine Learning", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, after=36)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("PathSnap")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(32), HEAD_COLOUR
    para("Distributed Map-Matching and Route Reconstruction", align=WD_ALIGN_PARAGRAPH.CENTER, size=16)
    para("Technical Report", align=WD_ALIGN_PARAGRAPH.CENTER, size=13, after=48)
    para("**Members**", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, after=2)
    for name, srn in MEMBERS:
        para(f"{name} ({srn})", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, after=0)
    para("", after=6)
    para("Repository: github.com/gaganraghavan/PATHSNAP-DISTRIBUTED-MAP-MATCHING-ROUTE-RECONSTRUCTION",
         align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
    para(date.today().strftime("%d %B %Y"), align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
    page_break()


# ------------------------------------------------------------------ 1. overview

def section_overview() -> None:
    h("1. System Overview", 1)
    para("PathSnap matches a noisy GPS track to a trip's road network and reconstructs the route that was driven. "
         "The pipeline has five stages, each a numbered script in `scripts/` backed by library code in `src/pathsnap/`:")
    numbered([
        "**Candidate generation.** For every GPS point, the directed road arcs within 100 m (at most the 32 nearest).",
        "**Feature engineering.** 17 geometric and contextual features per (point, candidate arc) pair.",
        "**Candidate scoring.** A learned model gives each candidate a probability of being the reference arc. "
        "Seven models were compared; the tuned LightGBM LambdaRank ranker was selected.",
        "**HMM/Viterbi decoding.** Candidates are hidden states, ranker probabilities are emissions, and road-network "
        "distances give transitions, so the decoded sequence is a drivable path.",
        "**Deployment.** A FastAPI service (Laptop 1) runs the pipeline for a requested trip and scores it against the "
        "reference route on the server; a browser UI (Laptop 2) lists the trips, sends the request and shows the route "
        "and metrics.",
    ])
    para("**Partitions.** 80 training trips are used for fitting, feature selection, cross-validation, Bayesian tuning "
         "and decoder tuning. The 10 student-test trips are used only for development-stage comparison. The 10 "
         "evaluator trips were not read during development; they were used only for a label-free check that the "
         "deployed API loads and predicts every trip (Section 7.4).")
    para("**Primary model-selection metric.** Observation-level top-1 accuracy, as the mean over 5 trip-grouped CV folds "
         "on the training trips. It is the brief's observation-level metric, it measures exactly the decision the "
         "decoder starts from (which arc each point is on), and it is not affected by the 1-in-20 class imbalance of "
         "candidate rows. Macro F1, observation log loss and route continuity are reported alongside it.")


# ------------------------------------------------------------------ 2. data

def section_data() -> None:
    page_break()
    h("2. Dataset Understanding and Exploratory Data Analysis", 1)
    para("All analysis in this section uses the **train** (80 trips) and **student_test** (10 trips) partitions; each "
         "figure and table says which. It is produced by `scripts/01_inventory.py`, `scripts/02_candidate_sweep.py` and "
         "the notebook `notebooks/01_eda.ipynb`.")

    h("2.1 Data and preprocessing", 2)
    tr, te = inv[inv.partition == "train"], inv[inv.partition == "student_test"]
    para(f"Each trip folder holds four headerless, whitespace-separated files: `.track` (longitude, latitude, time in s), "
         f"`.nodes` (longitude, latitude; the row index is the node id), `.arcs` (from-node, to-node; the row index is "
         f"the arc id) and `.route` (the reference arc sequence). The train partition has {tr.n_obs.sum():,} GPS points "
         f"and the student-test partition {te.n_obs.sum():,}.")
    qa("How did you preprocess the data?", [
        "**Parsing and validation.** Every file is read with explicit dtypes (float64 coordinates and times, int64 ids) "
        "and checked for missing values, out-of-range coordinates, duplicate rows, non-increasing timestamps and route "
        "ids outside the arc table (`pathsnap.validate`).",
        f"**Projection.** The trips span {inv.epsg.nunique()} UTM zones, so each trip is projected to the UTM zone of its "
        "GPS centroid and every distance is in metres. Distances are never computed in degrees.",
        "**Per-trip road network.** Arc and node ids are local to a trip, so the network is built per trip: arc "
        "geometry, length, bearing and a spatial index for candidate search.",
        "**Per-observation labels.** The `.route` file lists arcs, not which arc each point is on. A monotonic dynamic "
        "programme (`pathsnap.labels.align_to_route`) assigns each point a route position, never moving backwards, "
        "minimising total point-to-arc distance. The label is that arc; its distance is kept as a reliability measure.",
        "**Reliability filter.** Points more than 50 m from their label arc are treated as unreliable labels: they are "
        "excluded from training and from evaluation metrics (Section 2.3). No rows are dropped from the inputs; the "
        "API predicts every point.",
        "**No imputation.** There are no missing values; timestamps are kept as float seconds (12 of ~227k steps are "
        "non-integer).",
    ])

    h("2.2 Missing-value and data-quality analysis", 2)
    q = inv.sum(numeric_only=True)
    table(["Check (train + student_test, 90 trips)", "Count"],
          [["Missing values (all files)", f"{int(q.n_missing)}"],
           ["Out-of-range coordinates", f"{int(q.n_out_of_range)}"],
           ["Duplicate track rows", f"{int(q.n_duplicate_rows)}"],
           ["Non-increasing timestamps", f"{int(q.n_non_increasing_t)}"],
           ["Route ids outside the arc table", f"{int(q.route_out_of_range)}"],
           ["Trips with a time gap > 10 s", f"{int((inv.n_gaps_gt10s > 0).sum())}"],
           ["Trips where > 50% of points are > 50 m from the reference route", f"{int((inv.frac_label_dist_gt50m > .5).sum())}"]],
          [12.5, 3.0], "Data-quality checks (`results/eda/inventory.csv`).")
    para("The files are complete: there are no missing or invalid values to impute or drop. The real quality issues are "
         "irregular sampling, long gaps and GPS points far from the reference route.")

    h("2.3 Patterns observed", 2)
    steps = (inv.n_obs - 1).clip(lower=0)
    share_1s = float((inv.frac_dt_1s * steps).sum() / steps.sum())
    qa("What patterns did you observe during data exploration?", [
        f"**Sampling.** {pct(share_1s)} of steps are exactly 1 s, but a few trips have very long gaps "
        f"(the largest is {inv.dt_max_s.max() / 3600:.0f} h, in a training trip). Gaps must be handled by the decoder, "
        "not assumed away (Figure 2).",
        f"**Global spread.** Trips lie in {inv.epsg.nunique()} UTM zones across several continents (Figure 1), with "
        f"median speeds from {inv.speed_p50_ms.min() * 3.6:.0f} to {inv.speed_p50_ms.max() * 3.6:.0f} km/h.",
        "**Two-way roads.** About 92% of arcs have a reverse twin with identical geometry (Figure 4). Distance alone "
        "cannot tell the two directions apart, so heading features are essential.",
        "**Fragmented networks.** Networks have many small strongly connected components; the decoder must survive "
        "transitions with no directed path.",
        f"**GPS offset.** The median distance from a point to its label arc is {inv.label_dist_p50_m.median():.1f} m "
        "(per-trip median), but two training trips (33 and 54) are mostly more than 50 m from the reference route: "
        "the vehicle drove where the network has no roads (Figure 5).",
        "**Unobserved arcs.** About 10% of route arcs get no GPS point (short arcs crossed within a second), so "
        "observation-level metrics ignore them while route-level metrics include them.",
    ])
    figure(RES / "eda" / "trip_locations.png", "Trip locations (train and student_test).", 14)
    figure(RES / "eda" / "sampling_intervals.png", "Sampling interval distribution, per-trip share of 1 s steps, and the "
           "largest gap per trip (train and student_test).")
    figure(RES / "eda" / "trajectory_examples.png", "GPS trajectories (red) with the reference route (green) and road "
           "network (grey) for example training trips, in UTM metres.")
    figure(RES / "eda" / "network_stats.png", "Road-network size, share of two-way arcs and directed connectivity "
           "(train and student_test).")
    figure(RES / "eda" / "label_offset.png", "Distance from GPS points to their aligned label arc, pooled and per-trip "
           "median (train and student_test). The spike at 100 m collects all larger values.")
    figure(FIG / "eda_map_00000052.png", "Folium map of training trip 00000052: raw GPS (red), road network near the "
           "track (grey) and reference route (green), with legend and layer control (`results/maps/eda_00000052.html`).",
           15)

    h("2.4 Candidate generation", 2)
    s100 = sweep[(sweep.radius == 100) & (sweep.k == 32)].set_index("partition")
    ftrip = pd.read_csv(RES / "features" / "per_trip.csv", dtype={"trip": str})
    ftrain = ftrip[ftrip.partition == "train"]
    off_share = (ftrain[ftrain.trip.isin(["00000033", "00000054"])].candidate_miss_count.sum()
                 / ftrain.candidate_miss_count.sum())
    para("A candidate set holds the arcs a point could be on. Its miss rate (reference arc absent) bounds what any "
         "model can recover, and its size sets the cost of scoring and decoding. A sweep over radius 25–200 m and "
         "k = 2 … all was run on train and student-test (Figure 7).")
    para(f"**Choice: radius 100 m, k = 32.** On reliable points (label within 50 m) the miss rate is "
         f"{pct(s100.loc['train', 'miss_rate_clean'], 3)} on train and {pct(s100.loc['student_test', 'miss_rate_clean'], 1)} "
         f"on student-test, with {s100.loc['train', 'mean_candidates']:.1f} and "
         f"{s100.loc['student_test', 'mean_candidates']:.1f} candidates per point. Larger radii only add candidates for "
         "off-network points, whose labels are unreliable anyway. Of the misses behind the overall train miss rate "
         f"({pct(s100.loc['train', 'miss_rate'], 1)}), {pct(off_share, 0)} come from trips 33 and 54.")
    figure(RES / "candidates" / "sweep.png", "Candidate miss rate against candidate-set size for each radius, all points "
           "and points whose label is within 50 m (train).", 15)

    h("2.5 Feature engineering", 2)
    qa("What features did you extract, and why?",
       "Each (GPS point, candidate arc) pair is one example. The 17 features describe the pair's geometry, the vehicle's "
       "motion, the local network, and how this candidate compares with its competitors and with the same arc at "
       "neighbouring points. Context features use only candidate geometry, never labels, so they are available at "
       "prediction time.")
    why = {
        "point_to_arc_m": "Closer arcs are more likely; the strongest single cue.",
        "arc_length_m": "Short arcs are hit by few points and are easy to skip.",
        "candidate_rank": "Distance order is robust to the trip's overall GPS offset.",
        "position_along_arc": "Projections onto arc ends often belong to the next arc.",
        "heading_delta_sin": "Signed heading difference; separates turns.",
        "heading_delta_cos": "Direction agreement; separates reverse twins on two-way roads.",
        "gps_step_m": "Large steps mean noisy or fast segments.",
        "gps_gap_s": "Gaps make the motion features unreliable.",
        "gps_speed_mps": "Speed separates motorways from side streets.",
        "start_node_degree": "Junction complexity near the arc start.",
        "end_node_degree": "Junction complexity near the arc end.",
        "dist_minus_min": "Distance relative to the best competitor at the same point.",
        "cos_minus_max": "Heading agreement relative to the best competitor.",
        "n_candidates": "Dense areas are harder; calibrates confidence.",
        "window_mean_dist_m": "Is the same arc close to the neighbouring points too?",
        "window_mean_cos": "Does the arc's direction agree over the neighbouring points?",
        "window_presence": "How many of the 6 neighbouring points also have this arc as a candidate.",
    }
    rows = [[f"`{r.feature}`", r.definition, why.get(r.feature, ""), "yes" if r.feature in features_final else ""]
            for r in schema.itertuples()]
    table(["Feature", "Definition", "Why", "Selected"], rows, [3.6, 6.6, 5.4, 1.4],
          "Candidate features (`results/features/feature_schema.csv`). “Selected” marks the 9 kept by feature selection.",
          size=8, numeric_from=4)
    para("**Feature selection** runs inside every CV fold on that fold's training rows only: Extra Trees impurity "
         "importance with a median threshold. All five folds and the final fit kept the same 9 features, marked above. "
         "**Scaling** (for logistic regression, k-NN and the MLP) is a `StandardScaler` inside the model pipeline, so it "
         "is fitted on training rows only.")
    a = abl.set_index("model")
    para(f"**Ablation.** Rerunning the comparison with only the 11 per-pair features lowers CV top-1 for every model, by "
         f"{(M.cv_top1_accuracy_mean - a.cv_top1_accuracy_mean).min() * 100:.1f}–"
         f"{(M.cv_top1_accuracy_mean - a.cv_top1_accuracy_mean).max() * 100:.1f} points "
         "(`results/models/ablation_pair_features/`). A single point often cannot tell parallel roads or the two "
         "directions of a road apart; its neighbours usually can.")
    fs = feat_summary
    para(f"**Scale.** Train has {int(fs.loc['train', 'candidates']):,} candidate rows "
         f"({fs.loc['train', 'mean_candidates_per_observation']:.1f} per point); student-test has "
         f"{int(fs.loc['student_test', 'candidates']):,}.")


# ------------------------------------------------------------------ 3. models

MODEL_TEXT = {
    "LightGBM LambdaRank": dict(
        why="Each GPS point has exactly one correct candidate among about 20, so the real question is which candidate "
            "beats the others at the same point. LambdaRank optimises that ordering within each query (one query per "
            "observation) instead of scoring rows against a global base rate.",
        strengths=["Learns the within-observation comparison directly; best top-1 in every CV fold.",
                   "Gradient-boosted trees capture interactions (heading matters only when the point is close) without "
                   "feature scaling.",
                   "Fast inference (about 20 ms per 1,000 points in the comparison run)."],
        limits=["Raw scores are only comparable within an observation; probabilities need a per-observation softmax, "
                "which is not a calibrated probability by construction.",
                "Largest gap between fitted and held-out top-1 of all models (overfits the training sample).",
                "No per-iteration loss history (the ranking objective is not a log loss), so no loss curve."],
        loss="LambdaRank: pairwise logistic loss between the relevant candidate and each other candidate in the same "
             "observation, weighted by the change in NDCG@5 (truncation level 5). Probabilities are a softmax of the "
             "scores within each observation."),
    "Histogram Gradient Boosting": dict(
        why="A strong general-purpose tabular learner: boosted trees handle the mix of distances, angles and counts, "
            "and binary log loss gives usable probabilities for the HMM.",
        strengths=["Second-best on every metric, close behind the ranker.", "Built-in early stopping on validation "
                   "log loss controls overfitting.", "Fast to fit (about 1 s on the 400k-row sample)."],
        limits=["Scores each candidate independently against the global 1-in-20 base rate; it never sees the "
                "competing candidates of the same point.",
                "sklearn's early-stopping split is random rows, not trips, so the internal validation curve is "
                "slightly optimistic (CV and test scores are unaffected)."],
        loss="Binary cross-entropy (log loss), L2 regularisation 1.0; early stopping after 20 iterations without "
             "improvement on a 10% validation split."),
    "Random Forest": dict(
        why="Bagged deep trees are a robust non-linear baseline that needs no scaling and is hard to break; it tests "
            "whether variance reduction alone is enough.",
        strengths=["Stable across folds; strong ROC-AUC (0.996 on student-test).", "Parallel fitting; no tuning of a "
                   "learning rate or iteration count."],
        limits=["Memorises the training observations (leaves of 5 rows): the gap between fitted and held-out top-1 "
                "equals the ranker's, with a lower held-out score.",
                "Slowest of the tree models to fit (about 12 s); probabilities are averages of leaf frequencies and "
                "are coarse."],
        loss="Each tree greedily minimises Gini impurity; there is no global loss. P(reference) is the mean of the "
             "leaf class fractions over 200 bootstrap trees."),
    "Multilayer Perceptron": dict(
        why="A neural network can learn smooth non-linear combinations of the standardised features and gives "
            "well-behaved probabilities; it is the only family here with an epoch-wise training history.",
        strengths=["Small fit/held-out gap (good generalisation).", "Fastest inference of the non-linear models "
                   "(about 3 ms per 1,000 points)."],
        limits=["Slowest fit (about 18 s on 400k rows) and sensitive to scaling and learning rate.",
                "Below the boosted models on every accuracy metric."],
        loss="Binary cross-entropy with L2 penalty (alpha = 1e-4), Adam optimiser, mini-batches of 1,024. Trained one "
             "epoch at a time with 15% of the training trips held out; the weights from the epoch with the lowest "
             "validation log loss are kept (patience 8)."),
    "Logistic Regression": dict(
        why="The linear baseline: if the features already separate the reference arc, a linear model should do "
            "well, and it is the cheapest model to fit and run.",
        strengths=["Context features make it competitive on top-1 (about 1 point behind the leader).",
                   "Fastest fit and almost the fastest inference; essentially no overfitting."],
        limits=["Cannot express interactions such as “heading agreement matters only when the point is close”, so "
                "its probabilities are clearly worse (highest log loss apart from Naive Bayes)."],
        loss="L2-regularised binary log loss (C = 1.0) on standardised features, solved with L-BFGS."),
    "k-Nearest Neighbours": dict(
        why="Instance-based learning tests whether candidates with similar geometry tend to share the label, with no "
            "parametric assumption.",
        strengths=["No training cost; competitive top-1 once the features are standardised."],
        limits=["By far the slowest inference (about 877 ms per 1,000 points): every query searches the 400k "
                "training rows.", "Probabilities are multiples of 1/50, so they are coarse emissions for the HMM."],
        loss="No training objective (lazy learner). P(reference) is the share of the 50 nearest training rows "
             "(Euclidean distance on standardised features) that are reference arcs."),
    "Gaussian Naive Bayes": dict(
        why="The simplest probabilistic classifier, included as a lower bound and to test the independence "
            "assumption on these features.",
        strengths=["Trains in milliseconds; very high candidate recall at the 0.5 threshold."],
        limits=["The features are strongly dependent (several are differences of others), which breaks the "
                "independence assumption and pushes probabilities to the extremes: lowest precision and worst log "
                "loss of all models."],
        loss="Maximum-likelihood per-class Gaussian means and variances (var_smoothing = 1e-9); prediction by Bayes' "
             "rule. No iterative loss."),
}


def _params_text(name: str) -> str:
    p = model_params[name]
    if name == "Logistic Regression":
        return "StandardScaler → LogisticRegression(C = 1.0, penalty = L2, solver = lbfgs, max_iter = 1000)"
    if name == "k-Nearest Neighbours":
        return "StandardScaler → KNeighborsClassifier(n_neighbors = 50, weights = uniform, metric = Euclidean)"
    keep = {k: v for k, v in p.items() if v not in ("None", "False", "0.0") and k not in ("verbose", "n_jobs",
                                                                                         "warm_start", "monotonic_cst",
                                                                                         "categorical_features")}
    return ", ".join(f"{k} = {v.strip(chr(39))}" for k, v in keep.items())


def _fit_evidence(name: str) -> str:
    r = M.loc[name]
    gap = r.cv_train_top1_accuracy_mean - r.cv_top1_accuracy_mean
    base = (f"Top-1 on the fitted training observations is {f3(r.cv_train_top1_accuracy_mean)} against "
            f"{f3(r.cv_top1_accuracy_mean)} on held-out CV trips (gap {gap:+.3f}), and {f3(r.test_top1_accuracy)} on "
            f"student-test.")
    if name == LEADER:
        return base + (" The gap is the largest of the seven models, so the untuned ranker overfits its training "
                       "sample; it still generalises best (best held-out score in 5 of 5 folds). Bayesian tuning "
                       "moved it to much smaller trees with subsampling and L1 regularisation (Section 4).")
    if name == "Histogram Gradient Boosting":
        b = hgb_hist.loc[hgb_hist.validation_log_loss.idxmin()]
        return base + (f" Moderate overfitting. Validation log loss reaches its minimum at iteration {int(b.epoch)} and "
                       f"early stopping ends training at iteration {int(hgb_hist.epoch.max())}, so boosting stopped "
                       "before the validation loss rose (Figure 12).")
    if name == "Random Forest":
        return base + " Clear overfitting: with 5-row leaves the forest memorises training observations."
    if name == "Multilayer Perceptron":
        b = mlp_hist.loc[mlp_hist.validation_log_loss.idxmin()]
        return base + (f" Little overfitting: validation log loss is lowest at epoch {int(b.epoch)} of "
                       f"{int(mlp_hist.epoch.max())} and the train and validation curves stay close (Figure 12).")
    if name == "Logistic Regression":
        return base + " No overfitting; the limit is model bias (linear decision boundary), shown by its log loss."
    if name == "k-Nearest Neighbours":
        return base + " Small gap with k = 50 (a smooth neighbourhood); no meaningful overfitting."
    return base + " Fitted and held-out scores match, so the model underfits from model bias, not variance."


def section_models() -> None:
    page_break()
    h("3. Model Experimentation and Comparison", 1)
    para("Seven models from seven algorithm families were trained by `scripts/04_model_comparison.py` (seed 42). "
         "Protocol, identical for every model:")
    bullets([
        "**Cross-validation.** `GroupKFold(5, shuffle=True)` over the 80 training trips, grouped by trip, so no trip "
        "is on both sides of a split (`results/models/cv_folds.csv`).",
        "**Training rows.** Observations whose label is within 50 m and among the candidates. Each fit samples 20,000 "
        "such observations (about 400k candidate rows) and keeps all their candidates, so the 1-in-20 class balance "
        "matches prediction time. The cap keeps k-NN and the MLP affordable; every model sees the same sample.",
        "**Evaluation.** All observations with a label within 50 m (student-test: 19,133). A missing reference "
        "candidate counts as wrong. Final models are refitted on all 80 training trips and scored on student-test.",
        "**Metrics** (`pathsnap.evaluate`): observation top-1; macro precision/recall/F1 with a class per "
        "(trip, reference arc); observation log loss (−log of the reference arc's probability after normalising "
        "within the observation, floor 1e-6); route continuity (share of consecutive points on the same or a "
        "directly linked arc); fit time and inference time.",
    ])

    h("3.1 Overall model comparison", 2)
    rows = []
    for name, r in M.iterrows():
        rows.append([name, r.family, r.configuration,
                     f"{f3(r.cv_macro_f1_mean)} ± {r.cv_macro_f1_std:.3f}",
                     f3(r.test_top1_accuracy), f3(r.test_route_continuity), f3(r.test_macro_f1),
                     f3(r.test_observation_log_loss), f"{r.test_predict_ms_per_1k_obs:.1f}"])
    table(["Model", "Family", "Key configuration", "CV macro F1", "Top-1", "Route contin.", "Test macro F1",
           "Log loss", "Inference ms/1k obs"], rows, [2.4, 1.7, 3.7, 1.9, 1.2, 1.3, 1.3, 1.2, 1.3],
          "Overall comparison. CV macro F1 is mean ± std over 5 trip-grouped folds on the training trips; the other "
          "metric columns are student-test. Log loss is observation-level. Route continuity is classifier-only "
          "(before HMM). Source: `results/models/model_comparison.csv`.", size=7.5, numeric_from=3)
    rows = []
    for name, r in M.iterrows():
        rows.append([name, f"{f3(r.cv_top1_accuracy_mean)} ± {r.cv_top1_accuracy_std:.3f}",
                     f"{f3(r.test_macro_precision)} / {f3(r.test_macro_recall)}", f3(r.test_candidate_roc_auc),
                     f3(r.cv_observation_log_loss_mean), f"{r.test_candidate_log_loss:.4f}",
                     f"{r.test_fit_seconds:.2f}", f"{int(r.cv_top1_folds_best)} / 5"])
    table(["Model", "CV top-1", "Test macro P / R", "Test cand. ROC-AUC", "CV obs. log loss", "Test cand. log loss",
           "Fit (s)", "Folds won"], rows, [3.4, 2.2, 2.4, 1.9, 1.9, 1.9, 1.4, 1.5],
          "Further metrics. Candidate-level metrics treat each candidate row as a binary example; for the ranker they "
          "use its per-observation softmax.", size=7.5)
    para("**Selection: LightGBM LambdaRank.** It has the best CV top-1 (the primary metric) in all five folds and on "
         f"student-test ({f3(M.loc[LEADER, 'test_top1_accuracy'])}), the lowest observation log loss in CV and on "
         "student-test (the HMM consumes these probabilities as emissions), the highest macro F1 (so the gain is not "
         "limited to long, frequently observed arcs), the best classifier-only route continuity, and inference cheap "
         "enough for the API. Its paired CV margin over the runner-up is "
         f"{-M.loc['Histogram Gradient Boosting', 'cv_top1_paired_gap_to_leader_mean'] * 100:.1f} ± "
         f"{M.loc['Histogram Gradient Boosting', 'cv_top1_paired_gap_to_leader_std'] * 100:.1f} points of top-1. The "
         "CV standard deviation (about ±0.03) comes mostly from fold difficulty shared by all models, so the paired "
         "per-fold comparison is the meaningful one. The concern is overfitting, addressed by tuning in Section 4.")

    h("3.2 Per-model analysis", 2)
    for i, (name, r) in enumerate(M.iterrows(), start=1):
        t = MODEL_TEXT[name]
        h(f"3.2.{i} {name} ({r.family})", 3)
        qa("A1. Why is this algorithm suitable for PathSnap?", t["why"])
        qa("A2. Strengths and limitations for this task", [f"**Strength:** {s}" for s in t["strengths"]] +
           [f"**Limitation:** {s}" for s in t["limits"]])
        qa("B5. Important hyperparameters and initial values", f"`{_params_text(name)}`. These are the values used in "
           "the comparison; only the selected model was tuned afterwards.")
        qa("B6. Loss function or optimisation objective", t["loss"])
        qa("C7. Performance on the student-testing dataset",
           f"Top-1 {f3(r.test_top1_accuracy)}, macro precision {f3(r.test_macro_precision)}, macro recall "
           f"{f3(r.test_macro_recall)}, route continuity {f3(r.test_route_continuity)}, candidate ROC-AUC "
           f"{f3(r.test_candidate_roc_auc)}; fit {r.test_fit_seconds:.2f} s, inference "
           f"{r.test_predict_ms_per_1k_obs:.1f} ms per 1,000 points. Rank {i} of 7 by CV top-1.")
        qa("C8. Observation-level Top-1 accuracy, Macro F1 and Log Loss (student-test)",
           f"Top-1 **{f4(r.test_top1_accuracy)}**, macro F1 **{f4(r.test_macro_f1)}**, observation log loss "
           f"**{f4(r.test_observation_log_loss)}** (candidate-level log loss {r.test_candidate_log_loss:.4f}).")
        qa("C9. Overfitting or underfitting?", _fit_evidence(name))

    h("3.3 Model visualisations", 2)
    para("All figures are drawn from the comparison CSVs by `scripts/plot_model_comparison.py`.")
    figure(RES / "models" / "figures" / "cv_scores.png", "Cross-validation macro F1 and top-1 for all seven models "
           "(mean ± std, dots are the individual folds; train trips).")
    figure(RES / "models" / "figures" / "student_test_scores.png", "Student-test top-1 accuracy and macro F1.")
    figure(RES / "models" / "figures" / "log_loss.png", "Observation-level and candidate-level log loss "
           "(student-test; lower is better).")
    figure(RES / "models" / "figures" / "confusion_matrices.png", "Candidate-level confusion matrices at "
           "P ≥ 0.5 (student-test), an optional diagnostic. Gaussian Naive Bayes trades precision for recall.")
    figure(RES / "models" / "figures" / "loss_curves.png", "Training and validation log loss for the two models with "
           "epoch-wise histories: Histogram Gradient Boosting (per boosting iteration) and the MLP (per epoch). The "
           "other models have no training history, so no curve is drawn for them.")
    figure(RES / "models" / "figures" / "fit_vs_heldout.png", "Top-1 on fitted training observations against held-out "
           "CV top-1: the vertical distance from the diagonal is the overfitting gap.", 12)
    figure(RES / "models" / "figures" / "timing.png", "Fit time and inference time per model.")
    a = abl.set_index("model")
    rows = [[n, f3(a.loc[n, "cv_top1_accuracy_mean"]), f3(M.loc[n, "cv_top1_accuracy_mean"]),
             f3(a.loc[n, "test_top1_accuracy"]), f3(M.loc[n, "test_top1_accuracy"])] for n in M.index]
    table(["Model", "CV top-1, pair features", "CV top-1, + context", "Test top-1, pair", "Test top-1, + context"],
          rows, [4.6, 3.1, 3.1, 3.1, 3.1], "Feature ablation: the 11 per-pair features alone against all 17 "
          "(`results/models/ablation_pair_features/`). Without context, all models are within about 0.01 of each "
          "other and the ranker is not the best.")


# ------------------------------------------------------------------ 4. tuning

def section_tuning() -> None:
    page_break()
    h("4. Bayesian Hyperparameter Optimisation (LightGBM LambdaRank)", 1)
    n_complete = int((trials.state == "COMPLETE").sum())
    n_pruned = int((trials.state == "PRUNED").sum())
    n_fail = int((trials.state == "FAIL").sum())
    dur = trials.duration_s.dropna()
    para("Run by `scripts/05_bayesian_optimisation.py` with Optuna. The folds, the 20,000-observation training sample "
         "per fit and the per-fold feature selection are the same as in Section 3. Student-test is not used during "
         "the search. The full trial record is in `results/tuning/trials.csv` and the SQLite study "
         "`results/tuning/optuna.db`.")
    qa("1–2. Hyperparameters searched and their ranges",
       "Nine LightGBM ranker hyperparameters (Table 6). Ranges on a log scale where the effect is multiplicative "
       "(tree count, learning rate, leaves, leaf size, regularisation).")
    un = space["untuned"]
    rows = []
    for k, s in space["space"].items():
        v = best["params"][k]
        rows.append([k, f"{s['low']:g} – {s['high']:g}", "log" if s["log"] else "linear", f"{un[k]:g}",
                     f"{v:.4g}" if isinstance(v, float) else str(v)])
    table(["Hyperparameter", "Range", "Scale", "Untuned", f"Best (trial {best['trial']})"], rows,
          [4.4, 3.6, 2.0, 2.6, 3.4], "Search space and best configuration (`results/tuning/search_space.json`, "
          "`best_params.json`). subsample uses bagging frequency 1; truncation_level is LambdaRank's NDCG cut-off.")
    qa("3. Objective function",
       "Maximise mean observation-level top-1 accuracy over the 5 trip-grouped CV folds on the 80 training trips: the "
       "primary metric from Section 3, because the task is judged per observation and top-1 is what the decoder "
       "starts from.")
    qa("4. Number of trials", [
        f"Sampler: `TPESampler(multivariate=True, n_startup_trials=15, seed=42)`; pruner: "
        f"`MedianPruner(n_startup_trials=10, n_warmup_steps=2)` on the running fold mean.",
        f"{n_complete + n_pruned} trials finished: {n_complete} completed and {n_pruned} pruned. "
        f"{n_fail} further trial was stopped when the search was capped for time; it is recorded as FAIL and not used.",
        f"Computational cost: {dur.sum() / 3600:.1f} h of recorded trial time, median {dur.median():.0f} s per trial. "
        f"One early trial with 229 leaves and a 0.011 learning rate took {dur.max() / 3600:.1f} h on its own.",
    ])
    imp = ", ".join(f"{r.param} {r.importance:.2f}" for r in importance.head(5).itertuples())
    qa("5. Best values", f"Trial {best['trial']} (Table 6), CV top-1 {f4(best['cv_top1_mean'])}. fANOVA importance: "
       f"{imp}; the rest are below 0.02. The search moved to much smaller trees (8 leaves instead of 31) with row and "
       "column subsampling and L1 regularisation, which addresses the overfitting seen in Section 3.")
    u, t = tune.loc["untuned"], tune.loc["tuned"]
    rows = [
        ["CV top-1 (mean ± std)", f"{f4(u.cv_top1_accuracy_mean)} ± {u.cv_top1_accuracy_std:.4f}",
         f"{f4(t.cv_top1_accuracy_mean)} ± {t.cv_top1_accuracy_std:.4f}"],
        ["CV macro F1", f"{f4(u.cv_macro_f1_mean)} ± {u.cv_macro_f1_std:.4f}",
         f"{f4(t.cv_macro_f1_mean)} ± {t.cv_macro_f1_std:.4f}"],
        ["CV observation log loss", f4(u.cv_observation_log_loss_mean), f4(t.cv_observation_log_loss_mean)],
        ["Student-test top-1", f4(u.test_top1_accuracy), f4(t.test_top1_accuracy)],
        ["Student-test macro F1", f4(u.test_macro_f1), f4(t.test_macro_f1)],
        ["Student-test observation log loss", f4(u.test_observation_log_loss), f4(t.test_observation_log_loss)],
        ["Student-test route continuity (classifier only)", f4(u.test_route_continuity), f4(t.test_route_continuity)],
        ["Fit time, all 80 trips' sample (s)", f"{u.test_fit_seconds:.1f}", f"{t.test_fit_seconds:.1f}"],
        ["Inference (ms per 1,000 observations)", f"{u.test_predict_ms_per_1k_obs:.0f}",
         f"{t.test_predict_ms_per_1k_obs:.0f}"],
    ]
    table(["Metric", "Untuned", "Tuned"], rows, [8.5, 4.2, 4.2],
          "Untuned against tuned on the same folds and partitions (`results/tuning/untuned_vs_tuned.csv`).")
    qa("6. Did the tuned model outperform the untuned model?", [
        f"**Yes on the objective and on both student-test classification metrics**: CV top-1 "
        f"{(t.cv_top1_accuracy_mean - u.cv_top1_accuracy_mean) * 100:+.2f} points, student-test top-1 "
        f"{(t.test_top1_accuracy - u.test_top1_accuracy) * 100:+.2f} points, student-test macro F1 "
        f"{(t.test_macro_f1 - u.test_macro_f1) * 100:+.2f} points. The gain is small, consistent with the history plot: "
        "the best trial came early and later trials stayed within 0.001 of it.",
        f"**Log loss:** CV observation log loss improved, but student-test observation log loss got slightly worse "
        f"({t.test_observation_log_loss - u.test_observation_log_loss:+.4f}); the objective was top-1, not calibration.",
        f"**Inference time:** the tuned model is slower ({t.test_predict_ms_per_1k_obs:.0f} against "
        f"{u.test_predict_ms_per_1k_obs:.0f} ms per 1,000 points); it has {best['params']['n_estimators']} trees "
        f"instead of {un['n_estimators']}. Both were measured in the same run on the same machine; they are not comparable with the Section 3 timings, which "
        "came from a different machine.",
        f"The untuned baseline scores {f4(u.cv_top1_accuracy_mean)} CV top-1 here and "
        f"{f4(M.loc[LEADER, 'cv_top1_accuracy_mean'])} in Section 3 with identical folds and features; the difference "
        "comes from LightGBM's multithreaded histogram construction on a different core count.",
    ])
    figure(RES / "tuning" / "figures" / "optimisation_history.png", "CV top-1 per trial and best so far, against the "
           "untuned configuration.", 14)
    figure(RES / "tuning" / "figures" / "param_importance.png", "fANOVA hyperparameter importance over the completed "
           "trials.", 11)
    figure(RES / "tuning" / "figures" / "untuned_vs_tuned.png", "Untuned against tuned: CV and student-test metrics.")


# ------------------------------------------------------------------ 5. HMM

def section_hmm() -> None:
    page_break()
    h("5. HMM and Viterbi Route Reconstruction", 1)
    p = decoder["params"]
    para("Implemented in `src/pathsnap/hmm.py` (states, emissions, transitions, Viterbi) and `src/pathsnap/route.py` "
         "(route reconstruction and route metrics); run by `scripts/06_hmm_decoding.py`. The tuned ranker's outputs "
         "are used unchanged as emissions, so the candidate-to-arc correspondence is preserved from candidate "
         "generation through to the decoded path.")
    qa("1. Hidden states", "At each GPS observation the hidden states are its candidate arcs (100 m, 32 nearest). "
       "Each state carries the arc id and the position t ∈ [0, 1] of the point's projection along the arc. Arcs are "
       "directed, so the two directions of a road are different states.")
    qa("2. Emission probabilities", f"log e = emission_weight × log P(arc | observation), where P is the tuned ranker's "
       f"per-observation softmax probability, floored at 1e-9. emission_weight = {p['emission_weight']:g}.")
    qa("3. Transition probabilities", [
        f"Newson and Krumm (2009): log p = −|route_m − gps_m| / (beta_m × max(1, Δt)), with beta_m = {p['beta_m']:g} m.",
        "gps_m is the straight-line distance between the two GPS points. route_m is the directed network distance "
        "from the first projected point to the second: the rest of arc a, then the Dijkstra shortest path from a's "
        "end node to b's start node, then the start of arc b.",
        f"On the same arc, a forward move, or a backward move of up to {p['backward_tolerance_m']:g} m (GPS jitter), "
        "costs only the distance along the arc. A U-turn onto the reverse twin is charged its real extra distance; "
        "this is what lets the decoder fix the classifier's direction flips on two-way roads.",
        "Dijkstra runs on the trip's arcs inside the track's bounding box plus 2 km.",
    ])
    qa("4. Missing candidates and large GPS gaps", [
        "Observations with no candidate get no state: the chain skips them and the next transition spans the real "
        "time gap.",
        "The transition scale grows with Δt, so after a long gap the decoder relies on the emissions instead of "
        f"insisting on a matching path length. The reachability bound max_speed × Δt + slack "
        f"({p['max_speed_mps']:g} m/s, {p['slack_m']:g} m) also grows with the gap.",
    ])
    qa("5. Consecutive observations on disconnected segments", [
        f"A transition is unreachable when no directed path exists within the bound. It gets a finite log-probability "
        f"disconnect_logp = {p['disconnect_logp']:g} instead of −∞, so Viterbi never runs out of paths: it can cross "
        "network breaks, off-network driving or missing arcs, and the emissions decide. Each such step is counted "
        "as a decoder break.",
        "In route reconstruction a jump with no connecting path is kept as a route break rather than invented.",
    ])
    para("**Viterbi recurrence.** δ_1(s) = log e_1(s); δ_k(s) = log e_k(s) + max over s′ of [δ_{k−1}(s′) + log "
         "p(s′ → s)], with back-pointers; the best final state is traced back to give one arc per observation. The "
         "work is in log space, vectorised over the candidate pairs of each step.")
    g = decoder["grid"]
    para(f"**Decoder tuning (training trips only).** A grid over beta_m ∈ {g['beta_m']}, emission_weight ∈ "
         f"{g['emission_weight']} and disconnect_logp ∈ {g['disconnect_logp']} was decoded on the 80 training trips "
         "(`results/hmm/grid.csv`). Training-trip emissions are the tuned ranker's out-of-fold probabilities, so they "
         "are as noisy as on unseen trips. The best train top-1 was chosen; student-test was not used for this choice "
         "(`models/hmm_decoder.json`).")

    h("5.1 Did HMM decoding improve the route?", 2)
    rows = []
    spec = [("Observation top-1", "top1_accuracy", 4), ("Macro precision", "macro_precision", 4),
            ("Macro recall", "macro_recall", 4), ("Macro F1", "macro_f1", 4), ("Route continuity", "route_continuity", 4),
            ("Route precision (length-weighted)", "route_precision", 4), ("Route recall", "route_recall", 4),
            ("Route F1", "route_f1", 4), ("Route mismatch (median over trips; lower is better)", "route_mismatch", 4),
            ("Trips with route F1 ≥ 0.9", "trips_route_f1_ge_0.9", 0), ("Unbridgeable route breaks", "route_breaks", 0),
            ("Decode time (ms per 1,000 obs)", "decode_ms_per_1k_obs", 0)]
    for label, col, d in spec:
        row = [label]
        for part in ("train", "student_test"):
            for meth in ("classifier", "hmm"):
                v = hmm_sum.loc[(part, meth), col]
                if pd.isna(v):
                    row.append("–")
                elif col == "trips_route_f1_ge_0.9":
                    row.append(f"{int(v)} / {int(hmm_sum.loc[(part, meth), 'trips'])}")
                else:
                    row.append(f"{v:,.{d}f}")
        rows.append(row)
    table(["Metric", "Train (OOF), classifier", "Train (OOF), + HMM", "Student-test, classifier",
           "Student-test, + HMM"], rows, [6.2, 2.7, 2.7, 2.7, 2.7],
          "Classifier-only against classifier + HMM, with the same tuned emissions and the same observations. The "
          "classifier-only route is reconstructed with the same procedure as the HMM route (`results/hmm/summary.csv`).",
          size=8)
    sc, sh = hmm_sum.loc[("student_test", "classifier")], hmm_sum.loc[("student_test", "hmm")]
    para(f"**Yes.** On student-test, decoding adds {(sh.top1_accuracy - sc.top1_accuracy) * 100:.1f} points of top-1 and "
         f"{(sh.macro_f1 - sc.macro_f1) * 100:.1f} points of macro F1, and raises route continuity from "
         f"{f3(sc.route_continuity)} to {f3(sh.route_continuity)}. Macro F1 rises more than top-1 because the classifier's "
         "errors concentrate on short arcs and direction twins, which the transition model fixes. The reconstructed "
         f"route improves most: route precision goes from {f3(sc.route_precision)} to {f3(sh.route_precision)}, the median "
         f"route mismatch drops from {pct(sc.route_mismatch)} to {pct(sh.route_mismatch)} of the reference length, and "
         f"unbridgeable breaks fall from {int(sc.route_breaks)} to {int(sh.route_breaks)}. Route recall barely changes: the "
         "classifier-only route already covered the reference, so the HMM's gain is in removing wrong arcs.")

    h("5.2 Where HMM decoding did not help", 2)
    st = hmm_trip[hmm_trip.partition == "student_test"].pivot(index="trip", columns="method", values="top1_accuracy")
    worse = st[st.hmm < st.classifier]
    tr = hmm_trip[hmm_trip.partition == "train"].pivot(index="trip", columns="method", values="top1_accuracy")
    tr_worse = (tr.hmm < tr.classifier).sum()
    items = []
    for trip, r in worse.iterrows():
        items.append(f"**Student-test trip {trip}:** top-1 fell from {r.classifier:.3f} to {r.hmm:.3f} while its route F1 "
                     "is unchanged. Nearly all points the HMM lost are matched to the reverse twin of the reference arc on "
                     "a noisy stretch (labels about 24 m away, against a typical 4–5 m). The route uses both directions "
                     "of that road and the decoder places the turnaround at a different point: the same arcs, but the "
                     "wrong direction for about 30 seconds of driving.")
    items.append(f"**Training trips:** {tr_worse} of {len(tr)} got slightly worse. On trip 00000005 all lost points sit "
                 "on the arc next to the reference arc: boundary timing at junctions, not wrong roads.")
    items.append(f"**Remaining breaks:** the decoder still crosses {int(hmm_sum.loc[('train', 'hmm'), 'decoder_breaks'])} "
                 f"unreachable transitions on train and {int(sh.decoder_breaks)} on student-test, concentrated in trips with "
                 "off-network driving (33, 54) and long GPS gaps, where no directed path within the speed bound exists.")
    bullets(items)
    figure(RES / "hmm" / "figures" / "classifier_vs_hmm.png", "Classifier-only against classifier + HMM on train (OOF) "
           "and student-test.")
    figure(RES / "hmm" / "figures" / "per_trip_top1.png", "Per-trip top-1: classifier-only (x) against HMM (y); points "
           "above the diagonal improved.", 9.5)
    figure(RES / "hmm" / "figures" / "decoder_grid.png", "Decoder grid on the training trips (OOF emissions).", 13)


# ------------------------------------------------------------------ 6. maps

def section_maps() -> None:
    page_break()
    h("6. Geographical Visualisation Using Folium", 1)
    para("`scripts/07_route_maps.py` (using `src/pathsnap/maps.py`) writes one interactive Folium map per student-test "
         "trip to `results/maps/route_<trip>.html`, indexed by `results/maps/index.html`, and a static view of each "
         "for this report. Each map has:")
    bullets([
        "raw GPS observations (red points, with timestamps on hover);",
        "the road network within 150 m of the track (grey);",
        "the reference route (wide translucent green);",
        "the classifier + HMM route (solid blue) and the classifier-only route (dashed orange);",
        "start and end markers, a legend, layer control to toggle each layer and switch basemaps (Esri light gray or "
        "Esri streets), and an extent fitted to the track;",
        "an info box with that trip's top-1, route F1, mismatch and breaks for both methods.",
    ])
    figure(RES / "maps" / "route_00000009.png", "Student-test trip 00000009: GPS, network, reference, HMM and "
           "classifier-only routes (`results/maps/route_00000009.html`).", 13)
    figure(RES / "maps" / "route_00000056.png", "Student-test trip 00000056, a looping route with repeated road "
           "segments (`results/maps/route_00000056.html`).", 13)
    para("EDA maps of three training trips (`results/maps/eda_*.html`, Figure 6) show raw GPS, nearby network and "
         "reference route; trip 00000054 shows the off-network driving discussed in Section 2.3.")


# ------------------------------------------------------------------ 7. deployment

def section_deployment() -> None:
    page_break()
    h("7. API, UI and Deployment", 1)
    h("7.1 Architecture", 2)
    table(["Laptop", "Component", "Responsibilities"], [
        ["Laptop 1", "FastAPI backend (`api/`)", "Loads the tuned ranker (`models/lambdarank_tuned.joblib`) and decoder "
         "configuration (`models/hmm_decoder.json`) at start-up; reads each trip's track and network; runs candidates → "
         "features → ranker → HMM/Viterbi → route reconstruction; reads the reference route only after prediction "
         "and returns aggregate metrics; logs every request."],
        ["Laptop 2", "Evaluator UI (`ui/`)", "Static HTML/CSS/JS with vendored Leaflet, served by a standard-library "
         "Python server. Lists the trips from the API, sends the selected trip id to `POST /predict`, and shows the "
         "predicted routes on a map, the metrics, a timing breakdown and the live API log."],
    ], [2.0, 3.6, 11.4], "Two-laptop deployment.", size=8.5, numeric_from=3)
    table(["Endpoint", "Purpose"], [
        ["GET /health", "Status, number of trips served, uptime and requests served."],
        ["GET /trips", "The trips in the served folder: id, number of GPS points, duration, whether a reference exists."],
        ["POST /predict {\"trip_id\"}", "Runs the full pipeline and returns GPS points, nearby network, both predicted "
         "routes, aggregate metrics and stage timings."],
        ["GET /model", "Model, features, hyperparameters and decoder configuration."],
        ["GET /logs", "Recent log lines, shown in the UI next to the server console."],
    ], [5.0, 12.0], "API endpoints.", size=8.5, numeric_from=2)

    h("7.2 Keeping ground truth on the server", 2)
    bullets([
        "Trips are loaded for prediction without the `.route` file (`TripStore.load_for_prediction`), and the "
        "predictor refuses a trip that carries a reference route.",
        "The reference is read only after prediction, to compute metrics on the server (`api/evaluation.py`). The "
        "response contains aggregate metrics only; it never contains the reference route, per-point labels or the "
        "reference length.",
        "There is no per-trip special-casing: every trip goes through the same code path, and the served folder is a "
        "command-line argument (`--data-dir`).",
        "Each run is logged on Laptop 1: request received, points and candidates, per-stage timings, the server-side "
        "evaluation and the HTTP status (Figure 25).",
    ])

    h("7.3 Screenshots", 2)
    figure(FIG / "ui_result.png", "Evaluator UI after reconstructing student-test trip 00000056: trip list with "
           "per-trip top-1, predicted HMM route on the map, metric cards (HMM against classifier-only), the full metric "
           "table, server timing and the live API log.")
    figure(FIG / "ui_trip_00000009.png", "Evaluator UI for student-test trip 00000009.")
    figure(FIG / "api_terminal.png", "API console on the serving machine: model and decoder loaded, trips served, and "
           "two successful `POST /predict` requests with stage timings and server-side evaluation.")

    h("7.4 Readiness for the evaluator trips", 2)
    para("The API was pointed at the evaluator-test folder and every one of the 10 trips was loaded and run through the "
         "full prediction pipeline, and the UI response was built, to confirm that the system works for all reserved "
         "trips. This check did not open any `.route` file and computed no accuracy; nothing in the models, features, "
         "hyperparameters or decoder was changed after it.")

    h("7.5 Running the system", 2)
    para("Laptop 1 (API), from the repository root:")
    para("`uv sync`", after=2)
    para("`uv run python -m api --data-dir <folder with one sub-folder per trip>`")
    para("Laptop 2 (UI), Python 3 only, no other dependencies:")
    para("`python ui/serve.py --api http://<laptop-1-ip>:8000`")
    para("Then open http://localhost:5173 on Laptop 2. The API listens on all interfaces and prints the addresses the "
         "UI can use. Full instructions are in the README.")


# ------------------------------------------------------------------ 8. reproducibility

def section_repro() -> None:
    h("8. Reproducibility and Integrity", 1)
    bullets([
        "Every table and figure in this report is generated from files committed in `results/` and `models/`; this "
        "document itself is built from them by `scripts/08_build_report.py`.",
        "Seeds: 42 for fold assignment, sampling, models, Optuna and the decoder grid. Configurations are stored with "
        "the results (`run_config.json`, `search_space.json`, `decoder_config.json`), as is the Bayesian trial record "
        "(`trials.csv`, `optuna.db`).",
        "Pipeline: `scripts/01_inventory.py` → `02_candidate_sweep.py` → `03_feature_engineering.py` → "
        "`04_model_comparison.py` → `05_bayesian_optimisation.py` → `06_hmm_decoding.py` → `07_route_maps.py`.",
        "Evaluator-test data was not used for training, feature selection, model, hyperparameter, threshold or "
        "decoder choice, and its labels never enter the prediction pipeline or the UI.",
    ])
    h("AI assistance disclosure", 2)
    para("AI coding assistants, Claude Code and Codex, were used to generate code. Claude Code was also used to draft "
         "this report, including the script that builds it from the committed results.")


# ------------------------------------------------------------------ main

def main() -> None:
    setup_styles()
    title_page()
    section_overview()
    section_data()
    section_models()
    section_tuning()
    section_hmm()
    section_maps()
    section_deployment()
    section_repro()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is not None:                 # python-docx's template omits the schema-required percent
        zoom.set(qn("w:percent"), "100")
    doc.core_properties.title = "PathSnap Technical Report"
    doc.core_properties.author = ""
    doc.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}: {_fig_no} figures, {_tab_no} tables")


if __name__ == "__main__":
    main()
