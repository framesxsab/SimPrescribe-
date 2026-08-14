from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def safe_text(value: Any, fallback: str = "Not available") -> str:
    text = str(value or "").strip()
    return text or fallback


def safe_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def display_status(value: Any) -> str:
    return str(value or "needs_review").replace("_", " ").strip().title()


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
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(doc.leftMargin, 8 * mm, "Review aid only - verify against the original prescription")
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
        topMargin=22 * mm,
        bottomMargin=16 * mm,
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
        fontSize=15,
        leading=19,
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
        fontSize=7.8,
        leading=10,
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
    warning_style = ParagraphStyle(
        "Warning",
        parent=body_style,
        textColor=colors.HexColor("#7c2d12"),
        backColor=colors.HexColor("#fff7ed"),
        borderColor=colors.HexColor("#fdba74"),
        borderWidth=0.7,
        borderPadding=8,
        borderRadius=6,
    )
    direction_style = ParagraphStyle(
        "Direction",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=ink,
    )
    hero_style = ParagraphStyle(
        "Hero",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
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
    ocr_style = ParagraphStyle(
        "OCR",
        parent=body_style,
        fontName="Courier",
        fontSize=8.1,
        leading=11.2,
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
    patient_name = safe_text(analysis.get("patient_name"))
    doctor_name = safe_text(analysis.get("doctor_name"))
    prescription_date = safe_text(analysis.get("date"))
    pipeline = analysis.get("pipeline") if isinstance(analysis.get("pipeline"), dict) else {}
    ocr_confidence = pipeline.get("ocr_confidence")
    confidence_text = f"{float(ocr_confidence) * 100:.0f}%" if isinstance(ocr_confidence, (int, float)) else "Not reported"
    review_versions = analysis.get("review_versions") if isinstance(analysis.get("review_versions"), list) else []

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
                        [paragraph("PRESCRIPTION REVIEW BRIEF", chip_style)],
                        [paragraph("Prescription Analysis Report", hero_style)],
                        [
                            paragraph(
                                "Structured from OCR for review. Confirm every medication detail against the original prescription before use.",
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

    identity_table = build_detail_table(
        [
            [paragraph("Patient", meta_style), paragraph(patient_name, body_style), paragraph("Prescriber", meta_style), paragraph(doctor_name, body_style)],
            [paragraph("Prescription date", meta_style), paragraph(prescription_date, body_style), paragraph("Analysed", meta_style), paragraph(created_at, body_style)],
            [paragraph("Document", meta_style), paragraph(file_name, body_style), paragraph("OCR confidence", meta_style), paragraph(confidence_text, body_style)],
        ],
        col_widths=[26 * mm, 58 * mm, 26 * mm, 58 * mm],
        background="#f8fafc",
    )
    story.append(identity_table)
    story.append(Spacer(1, 10))
    story.append(
        paragraph(
            "This report is a review aid. Dosage, duration, and medication names should be validated with clinician or pharmacist guidance before any medicine is taken.",
            note_style,
        )
    )
    story.append(Spacer(1, 12))
    story.append(paragraph(f"Medication summary ({len(medications)})", title_style))
    story.append(paragraph("Read each direction beside the original prescription. Orange review boxes identify fields that need extra attention.", subheading_style))
    story.append(Spacer(1, 8))

    for index, med in enumerate(medications, start=1):
        medication_name = safe_text(med.get("name"), "Unknown medication")
        review_reasons = safe_list(med.get("review_reasons"))
        requires_review = bool(med.get("requires_review")) or bool(review_reasons)
        direction = " | ".join(
            [
                safe_text(med.get("dosage"), "Dose not captured"),
                safe_text(med.get("frequency"), "Frequency not captured"),
                safe_text(med.get("duration"), "Duration not captured"),
            ]
        )

        med_header = build_detail_table(
            [
                [paragraph(f"{index:02d}", chip_style), paragraph(medication_name, title_style), paragraph("REVIEW" if requires_review else "EXTRACTED", chip_style)],
                [paragraph("", body_style), paragraph(f"{safe_text(med.get('type'), 'Medication')} - {safe_text(med.get('category'), 'General')}", subheading_style), paragraph("", body_style)],
            ],
            col_widths=[12 * mm, 132 * mm, 24 * mm],
            background="#fff7ed" if requires_review else "#f0fdfa",
        )

        direction_table = build_detail_table(
            [[paragraph("PRESCRIBED DIRECTIONS", meta_style), paragraph(direction, direction_style)]],
            col_widths=[42 * mm, 126 * mm],
        )

        medication_card: list[Any] = [med_header, Spacer(1, 4), direction_table]
        if requires_review:
            review_text = " ".join(f"{position}. {reason}" for position, reason in enumerate(review_reasons, start=1)) or "Confirm this medication against the original prescription."
            medication_card.extend([Spacer(1, 4), paragraph(f"Needs review: {review_text}", warning_style)])
        story.append(KeepTogether(medication_card))

        detail_rows = []
        for label, value in (
            ("Composition", med.get("composition")),
            ("Manufacturer", med.get("manufacturer")),
            ("Pack size", med.get("pack_size")),
            ("Source", med.get("source")),
            ("Medication note", med.get("insight")),
            ("Common uses", ", ".join(safe_list(med.get("uses")))),
            ("Side effects", ", ".join(safe_list(med.get("side_effects"))[:8])),
            ("If unavailable - local dataset (not a switch instruction)", ", ".join(safe_list(med.get("substitutes"))[:6])),
            ("If unavailable - web/model (not a switch instruction)", ", ".join(str(item.get("name", "")) for item in med.get("web_alternatives", []) if isinstance(item, dict))),
            ("Reference classes", "; ".join(item for item in (med.get("therapeutic_class"), med.get("chemical_class"), med.get("action_class")) if isinstance(item, str) and item.strip())),
        ):
            if isinstance(value, str) and value.strip():
                detail_rows.append([paragraph(label, meta_style), paragraph(value.strip(), body_style)])
        if detail_rows:
            story.extend([Spacer(1, 4), build_detail_table(detail_rows, [38 * mm, 130 * mm], background="#f8fafc")])
        story.append(Spacer(1, 9))

    story.append(Spacer(1, 5))
    story.append(paragraph("Verification appendix", title_style))
    story.append(paragraph("Raw OCR extract", heading_style))
    story.append(paragraph("Compare this text directly with the source image. Line breaks are preserved because they can affect medication interpretation.", subheading_style))
    story.append(Spacer(1, 8))
    story.append(paragraph(raw_text or "No OCR text captured.", ocr_style))
    story.append(Spacer(1, 10))
    story.append(paragraph("Report trace", heading_style))
    story.append(build_detail_table([
        [paragraph("Report ID", meta_style), paragraph(report_id, body_style)],
        [paragraph("Dataset sources", meta_style), paragraph(", ".join(dataset_names) if dataset_names else "OCR only", body_style)],
        [paragraph("Review status", meta_style), paragraph(display_status(analysis.get("review_status")), body_style)],
        [paragraph("Prior review states", meta_style), paragraph(str(len(review_versions)), body_style)],
    ], col_widths=[38 * mm, 130 * mm], background="#f8fafc"))

    doc.build(story, onFirstPage=lambda canvas, report_doc: draw_page_chrome(canvas, report_doc, app_name), onLaterPages=lambda canvas, report_doc: draw_page_chrome(canvas, report_doc, app_name))
    return buffer.getvalue()
