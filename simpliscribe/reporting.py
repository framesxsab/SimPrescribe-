from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


def safe_text(value: Any, fallback: str = "Not available") -> str:
    text = str(value or "").strip()
    return text or fallback


def safe_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    escaped = (
        safe_text(text, "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return Paragraph(escaped, style)


def draw_page_chrome(canvas, doc, app_name: str) -> None:
    canvas.saveState()
    top_y = A4[1] - 13 * mm
    canvas.setFillColor(colors.HexColor("#0b6bcb"))
    canvas.roundRect(doc.leftMargin, top_y - 2 * mm, 11 * mm, 8 * mm, 2 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawCentredString(doc.leftMargin + 5.5 * mm, top_y + 0.2 * mm, "S")
    canvas.setFillColor(colors.HexColor("#0f172a"))
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(doc.leftMargin + 14 * mm, top_y, app_name)
    canvas.setStrokeColor(colors.HexColor("#dbe4e2"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, A4[1] - 16 * mm, A4[0] - doc.rightMargin, A4[1] - 16 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawRightString(A4[0] - doc.rightMargin, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_detail_table(rows: list[list[Paragraph]], col_widths: list[float], background: str = "#ffffff") -> Table:
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe4e2")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_pdf_report(analysis: dict[str, Any], app_name: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"{app_name} Prescription Report",
    )

    styles = getSampleStyleSheet()
    accent = colors.HexColor("#0b6bcb")
    ink = colors.HexColor("#0f172a")
    muted = colors.HexColor("#475569")
    panel = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=ink,
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=ink,
        spaceBefore=2,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        alignment=TA_LEFT,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=body_style,
        fontSize=8.4,
        leading=11,
        textColor=muted,
        textTransform="uppercase",
    )
    note_style = ParagraphStyle(
        "Note",
        parent=body_style,
        backColor=colors.HexColor("#eef6ff"),
        borderColor=colors.HexColor("#bfdbfe"),
        borderWidth=0.7,
        borderPadding=8,
        borderRadius=6,
    )
    hero_style = ParagraphStyle(
        "Hero",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=ink,
        spaceAfter=4,
    )
    subheading_style = ParagraphStyle(
        "Subheading",
        parent=body_style,
        fontSize=10,
        leading=13,
        textColor=muted,
    )
    chip_style = ParagraphStyle(
        "Chip",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        textColor=accent,
    )
    mono_style = ParagraphStyle(
        "Mono",
        parent=body_style,
        fontName="Courier",
        fontSize=8.3,
        leading=11,
        textColor=ink,
        backColor=panel,
        borderColor=colors.HexColor("#dbe4e2"),
        borderWidth=0.5,
        borderPadding=8,
        borderRadius=6,
    )

    medications = analysis.get("medications") if isinstance(analysis.get("medications"), list) else []
    dataset_names = sorted({name for med in medications for name in safe_list(med.get("source_datasets"))})
    file_name = safe_text(analysis.get("filename"), "Prescription Upload")
    report_id = safe_text(analysis.get("id") or analysis.get("analysis_id"))
    created_at = safe_text(analysis.get("created_at"))
    raw_text = safe_text(analysis.get("raw_text"), "No OCR text captured.")

    brand_panel = Table(
        [
            [
                Table(
                    [[paragraph("S", ParagraphStyle("LogoMark", parent=hero_style, alignment=1, textColor=colors.white, fontSize=22, leading=24))]],
                    colWidths=[14 * mm],
                    rowHeights=[14 * mm],
                ),
                Table(
                    [
                        [paragraph("Prescription Reading Aid", chip_style)],
                        [paragraph("Prescription Analysis Report", hero_style)],
                        [paragraph(app_name, title_style)],
                        [
                            paragraph(
                                "A structured summary generated from OCR extraction and medicine dataset enrichment. Confirm all details against the original prescription before use.",
                                subheading_style,
                            )
                        ],
                    ],
                    colWidths=[148 * mm],
                ),
            ]
        ],
        colWidths=[18 * mm, 148 * mm],
        hAlign="LEFT",
    )
    brand_panel.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#bfdbfe")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#0b6bcb")),
                ("BOX", (0, 0), (0, 0), 0, colors.HexColor("#0b6bcb")),
                ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ]
        )
    )

    story: list[Any] = []
    story.append(brand_panel)
    story.append(Spacer(1, 10))

    summary_table = build_detail_table(
        [
            [paragraph("Document", heading_style), paragraph(file_name, body_style)],
            [paragraph("Report ID", heading_style), paragraph(report_id, body_style)],
            [paragraph("Analysis Time", heading_style), paragraph(created_at, body_style)],
            [paragraph("Medication Count", heading_style), paragraph(str(len(medications)), body_style)],
            [paragraph("Datasets", heading_style), paragraph(", ".join(dataset_names) if dataset_names else "OCR only", body_style)],
        ],
        col_widths=[42 * mm, 126 * mm],
        background="#f8fafc",
    )
    story.append(summary_table)
    story.append(Spacer(1, 10))
    story.append(
        paragraph(
            "This report is a review aid. Dosage, duration, and medication names should be validated with clinician or pharmacist guidance before any medicine is taken.",
            note_style,
        )
    )
    story.append(Spacer(1, 12))

    for index, med in enumerate(medications, start=1):
        medication_name = safe_text(med.get("name"), "Unknown medication")

        med_header = build_detail_table(
            [
                [paragraph(f"Medication {index}", chip_style), paragraph(medication_name, heading_style)],
                [paragraph("Clinical Category", meta_style), paragraph(safe_text(med.get("category"), "General"), body_style)],
            ],
            col_widths=[32 * mm, 136 * mm],
            background="#ffffff",
        )

        med_table = build_detail_table(
            [
                [paragraph("Type", meta_style), paragraph(safe_text(med.get("type"), "Medication"), body_style), paragraph("Source", meta_style), paragraph(safe_text(med.get("source"), "OCR only"), body_style)],
                [paragraph("Dosage", meta_style), paragraph(safe_text(med.get("dosage"), "N/A"), body_style), paragraph("Frequency", meta_style), paragraph(safe_text(med.get("frequency"), "N/A"), body_style)],
                [paragraph("Duration", meta_style), paragraph(safe_text(med.get("duration"), "N/A"), body_style), paragraph("Pack Size", meta_style), paragraph(safe_text(med.get("pack_size")), body_style)],
                [paragraph("Composition", meta_style), paragraph(safe_text(med.get("composition")), body_style), paragraph("Manufacturer", meta_style), paragraph(safe_text(med.get("manufacturer")), body_style)],
                [paragraph("Therapeutic", meta_style), paragraph(safe_text(med.get("therapeutic_class")), body_style), paragraph("Chemical", meta_style), paragraph(safe_text(med.get("chemical_class")), body_style)],
                [paragraph("Action", meta_style), paragraph(safe_text(med.get("action_class")), body_style), paragraph("Dataset Sources", meta_style), paragraph(', '.join(safe_list(med.get("source_datasets"))) or 'OCR only', body_style)],
            ],
            col_widths=[24 * mm, 60 * mm, 24 * mm, 60 * mm],
        )

        reference_table = build_detail_table(
            [
                [paragraph("Medication Note", meta_style), paragraph(safe_text(med.get("insight"), "Follow the prescription exactly as provided."), body_style)],
                [paragraph("Substitutes", meta_style), paragraph(', '.join(safe_list(med.get("substitutes"))) or 'Not available', body_style)],
                [paragraph("Common Uses", meta_style), paragraph(', '.join(safe_list(med.get("uses"))) or 'Not available', body_style)],
                [paragraph("Side Effects", meta_style), paragraph(', '.join(safe_list(med.get("side_effects"))[:8]) or 'Not available', body_style)],
            ],
            col_widths=[32 * mm, 136 * mm],
            background="#f8fafc",
        )

        story.append(KeepTogether([med_header, Spacer(1, 5), med_table, Spacer(1, 5), reference_table, Spacer(1, 10)]))

    story.append(PageBreak())
    story.append(paragraph("OCR Extract", title_style))
    story.append(paragraph("The raw OCR output is included below so the structured summary can be cross-checked against the extracted text.", subheading_style))
    story.append(Spacer(1, 8))
    story.append(Preformatted(raw_text or "No OCR text captured.", mono_style))

    doc.build(story, onFirstPage=lambda canvas, report_doc: draw_page_chrome(canvas, report_doc, app_name), onLaterPages=lambda canvas, report_doc: draw_page_chrome(canvas, report_doc, app_name))
    return buffer.getvalue()