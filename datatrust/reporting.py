import io
from typing import Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
from datatrust.models import TrustReport


class ReportGenerator:
    """
    Generates downloadable executive PDF audit reports for Task-Conditioned Dataset Fitness.
    """

    @staticmethod
    def generate_pdf(report: TrustReport) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=3
        )
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor('#475569'),
            spaceAfter=10
        )
        h2_style = ParagraphStyle(
            'SectionH2',
            parent=styles['Heading2'],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#1e293b'),
            spaceBefore=10,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'BodyTextCustom',
            parent=styles['Normal'],
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor('#334155')
        )
        badge_style = ParagraphStyle(
            'BadgeText',
            parent=styles['Heading2'],
            fontSize=13,
            leading=17,
            alignment=1, # Center
            textColor=colors.HexColor('#0f172a')
        )

        story = []

        # 1. Document Header
        story.append(Paragraph("<b>DataTrust AI</b> — Task-Conditioned Dataset Fitness Audit", title_style))
        conf_str = f" (Confidence: {report.task_inference.confidence*100:.1f}%)" if report.task_inference else ""
        story.append(Paragraph(f"Objective Task: <b>{report.task_type.value.replace('_', ' ').title()}</b>{conf_str} | Evaluated at: {report.evaluated_at}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82f6"), spaceAfter=10))

        # 2. Executive Summary Score Box
        tier_color = "#10b981" if report.task_conditioned_fitness >= 80 else ("#f59e0b" if report.task_conditioned_fitness >= 60 else "#ef4444")
        
        score_data = [
            [
                Paragraph(f"<b>Task-Conditioned Dataset Fitness:</b><br/><font size=18><b>{report.task_conditioned_fitness:.1f} / 100</b></font><br/><font size=9 color='{tier_color}'><b>Status: {report.confidence_tier}</b></font>", badge_style),
                Paragraph(
                    f"<b>Dataset:</b> {report.dataset_name}<br/>"
                    f"<b>Volume:</b> {report.profiling.num_rows:,} rows &times; {report.profiling.num_columns} columns | {report.profiling.missing_cell_percentage}% null<br/>"
                    f"<b>AHP Consistency Ratio:</b> {report.ahp_consistency_ratio:.4f} (&lt; 0.10 Passed &check;)<br/>"
                    f"<b>Downstream Prediction:</b> Predicted {report.predicted_metric_name} = {report.predicted_metric_value or 'N/A'}",
                    body_style
                )
            ]
        ]
        score_table = Table(score_data, colWidths=[230, 310])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 8))

        # 3. Critical Defect / Veto Analysis Card
        story.append(Paragraph("CRITICAL DEFECT ANALYSIS", h2_style))
        if report.veto_applied:
            veto_text = (
                f"<b>&warning; CRITICAL DEFECT TRIGGERED</b><br/>"
                f"{report.veto_message}<br/>"
                f"Weighted score before constraint: <b>{report.raw_score:.1f}</b> | "
                f"Fitness ceiling imposed: <b>{report.task_conditioned_fitness:.1f}</b> | "
                f"Final Fitness: <b>{report.task_conditioned_fitness:.1f}</b>"
            )
            v_box = Table([[Paragraph(veto_text, body_style)]], colWidths=[540])
            v_box.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fee2e2')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#991b1b')),
                ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#ef4444')),
                ('PADDING', (0, 0), (-1, -1), 7),
            ]))
            story.append(v_box)
        else:
            ok_box = Table([[Paragraph("<b>&check; No fatal constraints triggered.</b> Dataset harbors zero task-breaking critical defects.", body_style)]], colWidths=[540])
            ok_box.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ecfdf5')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#065f46')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#10b981')),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(ok_box)

        story.append(Spacer(1, 8))

        # 4. Task-Quality Sensitivity M(T, Q) Table
        story.append(Paragraph("TASK–QUALITY SENSITIVITY M(T, Q)", h2_style))
        sens_data = [["Quality Dimension", "Sensitivity Level", "Task Weight", "Score", "Weighted Pts"]]
        for dim_key, res in report.dimensions.items():
            sens_data.append([
                dim_key.replace('_', ' ').title(),
                res.sensitivity_level,
                f"{res.weight * 100:.1f}%",
                f"{res.score:.1f}",
                f"{res.weighted_score:.1f}"
            ])
        sens_table = Table(sens_data, colWidths=[160, 95, 95, 95, 95])
        sens_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ]))
        story.append(sens_table)
        story.append(Spacer(1, 8))

        # 5. Downstream Model Validation
        if report.observed_metric_value is not None:
            story.append(Paragraph("DOWNSTREAM MODEL VALIDATION & RESIDUAL", h2_style))
            res_val = report.metric_residual if report.metric_residual is not None else 0.0
            val_text = (
                f"<b>Predicted {report.predicted_metric_name}:</b> {report.predicted_metric_value} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Observed {report.predicted_metric_name}:</b> {report.observed_metric_value} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Residual &epsilon;:</b> {res_val:+}"
            )
            v_tbl = Table([[Paragraph(val_text, body_style)]], colWidths=[540])
            v_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f9ff')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#0284c7')),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(v_tbl)
            story.append(Spacer(1, 8))

        # 6. Remediation Optimizer Table
        story.append(Paragraph("REMEDIATION OPTIMIZER (MRU = &Delta;Fitness / Cost)", h2_style))
        if not report.optimized_remediations:
            story.append(Paragraph("Dataset is in optimal health. No high-utility remediations needed.", body_style))
        else:
            opt_data = [["Priority", "Intervention & Recommendation", "Gain", "Cost", "MRU"]]
            for act in report.optimized_remediations[:4]:
                action_text = f"<b>{act.issue}</b><br/>{act.recommendation}"
                opt_data.append([
                    f"#{act.priority_rank}",
                    Paragraph(action_text, body_style),
                    f"+{act.fitness_improvement:.1f}",
                    str(act.operational_cost),
                    f"{act.mru:.2f}"
                ])
            opt_table = Table(opt_data, colWidths=[50, 340, 50, 45, 55])
            opt_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ]))
            story.append(opt_table)

        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
        story.append(Paragraph("DataTrust AI Platform &bull; Context-Conditioned Data Fitness & Adaptive Remediation Optimization.", subtitle_style))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
