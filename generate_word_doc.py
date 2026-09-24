"""
Script to generate a highly professional Microsoft Word (.docx) document
from the CardioSense AI Project and Viva Guide.
"""

import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set cell internal margins (padding)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_styled_document():
    doc = Document()

    # Set Page Margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)  # Slate 800

    # Read Markdown File
    md_path = os.path.join(os.path.dirname(__file__), "PROJECT_AND_VIVA_GUIDE.md")
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Title Banner
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("🩺 CardioSense AI\n")
    title_run.font.name = 'Segoe UI'
    title_run.font.size = Pt(24)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)  # Ocean Blue

    sub_run = title_p.add_run("Complete Project Documentation & Viva Master Guide")
    sub_run.font.name = 'Segoe UI'
    sub_run.font.size = Pt(14)
    sub_run.font.bold = True
    sub_run.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)

    badge_p = doc.add_paragraph()
    badge_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    b_run = badge_p.add_run("Healthcare Informatics • Applied Machine Learning • Clinical Decision Support System")
    b_run.font.size = Pt(9.5)
    b_run.font.italic = True
    b_run.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Parse lines
    lines = content.split("\n")
    i = 0
    in_table = False
    table_lines = []

    def render_markdown_table(tbl_lines):
        if not tbl_lines or len(tbl_lines) < 2:
            return
        # Parse rows
        raw_rows = []
        for tl in tbl_lines:
            if not tl.strip().startswith("|"):
                continue
            cells = [c.strip() for c in tl.strip().split("|")[1:-1]]
            if cells and not all(re.match(r'^:?-+:?$', c) for c in cells):
                raw_rows.append(cells)

        if not raw_rows:
            return

        cols_count = max(len(r) for r in raw_rows)
        tbl = doc.add_table(rows=len(raw_rows), cols=cols_count)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = True

        for r_idx, row_data in enumerate(raw_rows):
            for c_idx in range(cols_count):
                cell_text = row_data[c_idx] if c_idx < len(row_data) else ""
                cell = tbl.cell(r_idx, c_idx)
                cell.text = ""
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.space_after = Pt(3)

                if r_idx == 0:
                    set_cell_background(cell, "0284C7")  # Header Blue
                    run = p.add_run(cell_text.replace("**", "").replace("*", ""))
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.font.size = Pt(9.5)
                else:
                    if r_idx % 2 == 1:
                        set_cell_background(cell, "F8FAFC")
                    else:
                        set_cell_background(cell, "FFFFFF")
                    
                    # Parse bold markers in cell
                    parts = re.split(r'(\*\*.*?\*\*)', cell_text)
                    for pt in parts:
                        if pt.startswith("**") and pt.endswith("**"):
                            r = p.add_run(pt[2:-2])
                            r.font.bold = True
                            r.font.size = Pt(9.5)
                            r.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
                        else:
                            r = p.add_run(pt)
                            r.font.size = Pt(9.5)
                            r.font.color.rgb = RGBColor(0x37, 0x41, 0x51)

                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    while i < len(lines):
        line = lines[i]

        # Check for table
        if line.strip().startswith("|"):
            table_lines.append(line)
            in_table = True
            i += 1
            continue
        elif in_table:
            render_markdown_table(table_lines)
            table_lines = []
            in_table = False

        # Header 1 (e.g. # 1. Section)
        if line.startswith("# ") and not line.startswith("# 🩺"):
            heading_text = line[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(heading_text)
            run.font.name = 'Segoe UI'
            run.font.size = Pt(18)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)

        # Header 2 (e.g. ## 2. Section)
        elif line.startswith("## "):
            heading_text = line[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(heading_text)
            run.font.name = 'Segoe UI'
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)  # Dark Slate

        # Header 3 (e.g. ### 3. Section or ### Q1:)
        elif line.startswith("### "):
            heading_text = line[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(2)
            
            # Special highlighting for Viva Questions
            if heading_text.startswith("Q") and ":" in heading_text:
                run = p.add_run("🎯 " + heading_text)
                run.font.name = 'Segoe UI'
                run.font.size = Pt(11.5)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0x1D, 0x4E, 0xD8)  # Royal Blue
            else:
                run = p.add_run(heading_text)
                run.font.name = 'Segoe UI'
                run.font.size = Pt(12)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

        # Header 4
        elif line.startswith("#### "):
            heading_text = line[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(heading_text)
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

        # Horizontal rule
        elif line.strip() == "---":
            pass

        # Viva Answer Box or Regular block
        elif line.startswith("**Answer:**"):
            ans_p = doc.add_paragraph()
            ans_p.paragraph_format.left_indent = Inches(0.2)
            ans_p.paragraph_format.space_after = Pt(6)
            
            ans_run = ans_p.add_run("💡 Answer: ")
            ans_run.font.bold = True
            ans_run.font.color.rgb = RGBColor(0x05, 0x96, 0x69)  # Emerald green
            
            rest = line.replace("**Answer:**", "").strip()
            parts = re.split(r'(\*\*.*?\*\*|\$.*?\$|`.*?`)', rest)
            for pt in parts:
                if pt.startswith("**") and pt.endswith("**"):
                    r = ans_p.add_run(pt[2:-2])
                    r.font.bold = True
                elif pt.startswith("`") and pt.endswith("`"):
                    r = ans_p.add_run(pt[1:-1])
                    r.font.name = 'Consolas'
                    r.font.size = Pt(10)
                    r.font.color.rgb = RGBColor(0x7C, 0x3A, 0xED)
                else:
                    ans_p.add_run(pt.replace("$", ""))

        # Bullet items
        elif line.strip().startswith("- ") or line.strip().startswith("* "):
            bullet_p = doc.add_paragraph(style='List Bullet')
            bullet_p.paragraph_format.space_after = Pt(2)
            item_text = line.strip()[2:].strip()
            
            parts = re.split(r'(\*\*.*?\*\*|`.*?`)', item_text)
            for pt in parts:
                if pt.startswith("**") and pt.endswith("**"):
                    r = bullet_p.add_run(pt[2:-2])
                    r.font.bold = True
                elif pt.startswith("`") and pt.endswith("`"):
                    r = bullet_p.add_run(pt[1:-1])
                    r.font.name = 'Consolas'
                    r.font.size = Pt(10)
                else:
                    bullet_p.add_run(pt)

        # Numbered items
        elif re.match(r'^\d+\.\s', line.strip()):
            num_p = doc.add_paragraph(style='List Number')
            num_p.paragraph_format.space_after = Pt(2)
            item_text = re.sub(r'^\d+\.\s', '', line.strip())
            
            parts = re.split(r'(\*\*.*?\*\*|`.*?`)', item_text)
            for pt in parts:
                if pt.startswith("**") and pt.endswith("**"):
                    r = num_p.add_run(pt[2:-2])
                    r.font.bold = True
                elif pt.startswith("`") and pt.endswith("`"):
                    r = num_p.add_run(pt[1:-1])
                    r.font.name = 'Consolas'
                    r.font.size = Pt(10)
                else:
                    num_p.add_run(pt)

        # Standard Paragraph
        elif line.strip():
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            parts = re.split(r'(\*\*.*?\*\*|`.*?`)', line)
            for pt in parts:
                if pt.startswith("**") and pt.endswith("**"):
                    r = p.add_run(pt[2:-2])
                    r.font.bold = True
                elif pt.startswith("`") and pt.endswith("`"):
                    r = p.add_run(pt[1:-1])
                    r.font.name = 'Consolas'
                    r.font.size = Pt(10)
                    r.font.color.rgb = RGBColor(0x7C, 0x3A, 0xED)
                else:
                    p.add_run(pt)

        i += 1

    # Render any trailing table
    if in_table and table_lines:
        render_markdown_table(table_lines)

    output_path = os.path.join(os.path.dirname(__file__), "CardioSense_AI_Project_and_Viva_Guide.docx")
    doc.save(output_path)
    print(f"[SUCCESS] Created Word document: {output_path}")

    # Also save as PROJECT_AND_VIVA_GUIDE.docx for convenience
    doc.save(os.path.join(os.path.dirname(__file__), "PROJECT_AND_VIVA_GUIDE.docx"))

if __name__ == "__main__":
    create_styled_document()
