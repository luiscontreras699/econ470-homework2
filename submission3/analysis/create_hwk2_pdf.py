from pathlib import Path
import pandas as pd

# ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Pillow (for image size)
from PIL import Image as PILImage


# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "submission3" / "results"
OUT_PDF = RESULTS / "contreras-l-hwk2-3.pdf"

# Your actual filenames (from your screenshot)
FIG_TASK1 = RESULTS / "task1_boxplot.png"
FIG_T2_2014 = RESULTS / "task2_bid_hist_2014.png"
FIG_T2_2018 = RESULTS / "task2_bid_hist_2018.png"
FIG_T3 = RESULTS / "task3_hhi_trend.png"
FIG_T4 = RESULTS / "task4_ma_share_trend.png"

CSV_T5 = RESULTS / "task5_avg_bid_by_competition.csv"
CSV_T6 = RESULTS / "task6_avg_bid_by_ffs_quartile.csv"
CSV_T7 = RESULTS / "task7_ate_results.csv"


# -----------------------------
# Fonts (Times New Roman)
# -----------------------------
def register_times_new_roman():
    """
    Tries to register Times New Roman from common Windows font paths.
    Falls back to built-in Times-Roman if not found.
    """
    candidates = [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("C:/Windows/Fonts/timesi.ttf"),
        Path("C:/Windows/Fonts/timesbi.ttf"),
        # Some systems use different names:
        Path("C:/Windows/Fonts/timesnewroman.ttf"),
        Path("C:/Windows/Fonts/timesnewromanps.ttf"),
    ]

    # Prefer standard Windows times.ttf set
    if candidates[0].exists():
        pdfmetrics.registerFont(TTFont("TimesNewRoman", str(candidates[0])))
        if candidates[1].exists():
            pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", str(candidates[1])))
        if candidates[2].exists():
            pdfmetrics.registerFont(TTFont("TimesNewRoman-Italic", str(candidates[2])))
        if candidates[3].exists():
            pdfmetrics.registerFont(TTFont("TimesNewRoman-BoldItalic", str(candidates[3])))
        return True

    # Try any candidate that exists as regular
    for p in candidates:
        if p.exists():
            pdfmetrics.registerFont(TTFont("TimesNewRoman", str(p)))
            return True

    return False


# -----------------------------
# Helpers
# -----------------------------
def file_exists_or_warn(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_csv_table(path: Path):
    file_exists_or_warn(path)
    return pd.read_csv(path)


def df_to_reportlab_table(df: pd.DataFrame, max_rows=50):
    """
    Convert a dataframe to a simple ReportLab Table.
    Limits rows to max_rows to prevent overflow.
    """
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)

    # Round numeric columns for readability
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].round(3)

    data = [list(d.columns)] + d.astype(str).values.tolist()
    return data


def add_figure(flow, img_path: Path, caption: str,
               max_width=6.5 * inch, max_height=7.2 * inch):
    """
    Add an image scaled to fit within (max_width, max_height) while preserving aspect ratio.
    Wrap image+caption in KeepTogether so it doesn't get split/clipped.
    """
    file_exists_or_warn(img_path)

    # Read pixel size using Pillow
    with PILImage.open(img_path) as im:
        w_px, h_px = im.size

    aspect = h_px / w_px  # height/width

    # Scale based on width first
    draw_w = max_width
    draw_h = draw_w * aspect

    # If too tall, scale based on max_height instead
    if draw_h > max_height:
        draw_h = max_height
        draw_w = draw_h / aspect

    img = Image(str(img_path))
    img.drawWidth = draw_w
    img.drawHeight = draw_h

    block = [
        img,
        Spacer(1, 0.08 * inch),
        Paragraph(caption, STYLES["Caption"]),
        Spacer(1, 0.2 * inch),
    ]
    flow.append(KeepTogether(block))


def add_table(flow, df: pd.DataFrame, caption: str):
    data = df_to_reportlab_table(df)
    t = Table(data, hAlign="LEFT")

    style = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), BODY_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
    ])
    t.setStyle(style)

    flow.append(t)
    flow.append(Spacer(1, 0.08 * inch))
    flow.append(Paragraph(caption, STYLES["Caption"]))
    flow.append(Spacer(1, 0.25 * inch))


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY_FONT, 10)
    canvas.drawRightString(7.5 * inch, 0.6 * inch, f"Page {doc.page}")
    canvas.restoreState()


# -----------------------------
# Build PDF
# -----------------------------
if __name__ == "__main__":
    has_tnr = register_times_new_roman()
    BODY_FONT = "TimesNewRoman" if has_tnr else "Times-Roman"

    styles = getSampleStyleSheet()
    STYLES = {
        "Title": ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontName=BODY_FONT,
            fontSize=18,
            leading=22,
            spaceAfter=12,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontName=BODY_FONT,
            fontSize=12,
            leading=14,
            spaceAfter=12,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName=BODY_FONT,
            fontSize=14,
            leading=18,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName=BODY_FONT,
            fontSize=12,
            leading=15,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName=BODY_FONT,
            fontSize=12,
            leading=15,
            spaceAfter=8,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontName=BODY_FONT,
            fontSize=10,
            leading=12,
            textColor=colors.black,
            spaceAfter=6,
        ),
    }

    # Check required files
    for p in [FIG_TASK1, FIG_T2_2014, FIG_T2_2018, FIG_T3, FIG_T4, CSV_T5, CSV_T6, CSV_T7]:
        file_exists_or_warn(p)

    # Load tables
    t5 = load_csv_table(CSV_T5)
    t6 = load_csv_table(CSV_T6)
    t7 = load_csv_table(CSV_T7)

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        rightMargin=1 * inch,
        leftMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=0.9 * inch,
        title="Homework 2",
        author="Luis Contreras",
    )

    flow = []

    # Title header
    flow.append(Paragraph("Homework 2", STYLES["Title"]))
    flow.append(Paragraph("ECON 470 — Research Methods, Spring 2026", STYLES["Subtitle"]))
    flow.append(Paragraph("Luis Contreras", STYLES["Subtitle"]))
    flow.append(Spacer(1, 0.2 * inch))

    # ---- Summarize ----
    flow.append(Paragraph("Summarize the data (2014–2019)", STYLES["H1"]))

    # Task 1
    flow.append(Paragraph("1. Plan counts by county over time", STYLES["H2"]))
    flow.append(Paragraph(
        "After removing SNPs, 800-series plans, and prescription-drug-only plans, "
        "the distribution of plan counts varies across counties and years.",
        STYLES["Body"]
    ))
    add_figure(flow, FIG_TASK1, "Figure 1. Distribution of plan counts by county over time.")
    flow.append(Paragraph(
        "Interpretation: In most counties there are multiple plan options, but some counties have fewer. "
        "Overall, plan availability looks generally sufficient, with some areas showing limited choice.",
        STYLES["Body"]
    ))

    # Task 2
    flow.append(Paragraph("2. Bid distributions (2014 vs 2018)", STYLES["H2"]))
    # Put each histogram on its own page so it never clips
    flow.append(PageBreak())
    add_figure(flow, FIG_T2_2014, "Figure 2A. Bid distribution (2014).", max_height=7.2 * inch)

    flow.append(PageBreak())
    add_figure(flow, FIG_T2_2018, "Figure 2B. Bid distribution (2018).", max_height=7.2 * inch)

    flow.append(Paragraph(
        "Interpretation: The 2018 distribution is more spread out, with a heavier right tail, "
        "while 2014 is more concentrated.",
        STYLES["Body"]
    ))

    # Task 3
    flow.append(PageBreak())
    flow.append(Paragraph("3. Average HHI over time", STYLES["H2"]))
    add_figure(flow, FIG_T3, "Figure 3. Average HHI over time (2014–2019).", max_height=7.2 * inch)
    flow.append(Paragraph(
        "Interpretation: Average HHI moves modestly over time, suggesting competition changes year-to-year.",
        STYLES["Body"]
    ))

    # Task 4
    flow.append(PageBreak())
    flow.append(Paragraph("4. Medicare Advantage share over time", STYLES["H2"]))
    add_figure(flow, FIG_T4, "Figure 4. Average Medicare Advantage share over time (2014–2019).", max_height=7.2 * inch)
    flow.append(Paragraph(
        "Interpretation: MA share increases over time, indicating Medicare Advantage became more popular.",
        STYLES["Body"]
    ))

    flow.append(PageBreak())

    # ---- ATE section ----
    flow.append(Paragraph("Estimate ATEs (2018 only)", STYLES["H1"]))
    flow.append(Paragraph(
        "Competitive markets are counties in the bottom 33rd percentile of HHI, and uncompetitive markets "
        "are counties in the top 66th percentile of HHI.",
        STYLES["Body"]
    ))

    # Task 5
    flow.append(Paragraph("5. Average bid in competitive vs uncompetitive markets", STYLES["H2"]))
    add_table(flow, t5, "Table 5. Average bid (proxy) by competitive vs uncompetitive counties (2018).")

    # Task 6
    flow.append(Paragraph("6. Average bid by FFS-cost quartile and treatment/control", STYLES["H2"]))
    add_table(flow, t6, "Table 6. Average bid (proxy) by treatment/control within FFS quartiles (2018).")

    # Task 7
    flow.append(Paragraph("7. ATE estimates using different estimators", STYLES["H2"]))
    add_table(flow, t7, "Table 7. ATE estimates using matching, IPW, and regression (2018).")

    # Task 8–10
    flow.append(Paragraph("8. Are the estimators similar?", STYLES["H2"]))
    flow.append(Paragraph(
        "They are similar in sign but not identical in size. IPW and regression are almost the same, "
        "while matching varies more depending on the distance metric.",
        STYLES["Body"]
    ))

    flow.append(Paragraph("9. Preferred estimator with continuous covariates", STYLES["H2"]))
    flow.append(Paragraph(
        "Using regression as my preferred method, the estimate stays negative and similar in size when "
        "switching from quartiles to continuous covariates. The conclusion does not change.",
        STYLES["Body"]
    ))

    flow.append(Paragraph("10. Reflection", STYLES["H2"]))
    flow.append(Paragraph(
        "I learned how important identifiers and merges are in applied work. The most frustrating part was "
        "how long it took to download and organize all the files from OneDrive, especially with inconsistent formats.",
        STYLES["Body"]
    ))

    doc.build(flow, onFirstPage=header_footer, onLaterPages=header_footer)

    print(f"Saved PDF to: {OUT_PDF}")
    if not has_tnr:
        print("NOTE: Times New Roman font not found — used built-in Times-Roman instead.")