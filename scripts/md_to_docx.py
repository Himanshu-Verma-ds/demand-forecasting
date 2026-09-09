from __future__ import annotations

"""Convert a Markdown report to .docx without dropping content.

Written rather than shelling out to pandoc because pandoc is not installed in this environment.
Handles the constructs the report actually uses: ATX headings, pipe tables, fenced code blocks,
bullet and numbered lists, blockquotes, horizontal rules, and inline bold / italic / code / links.

    python scripts/md_to_docx.py final_conclusion/final_report.md final_conclusion/final_report.docx
"""

import argparse
import re
from pathlib import Path

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

CODE_FONT = "Consolas"
BODY_FONT = "Calibri"

# Everything prints black: this is a document to be read and marked up, not a web page.
BLACK = RGBColor(0x00, 0x00, 0x00)
LINK_BLUE = "0563C1"    # Office default hyperlink blue; the one non-black element
HEADER_FILL = "F2F2F2"  # neutral grey for table headers, not a colour accent

# Inline: `code`, **bold**, *italic*, [text](url). Order matters - code first so its
# contents are never re-parsed as emphasis.
INLINE = re.compile(
    r"(`[^`]+`)"
    r"|(\*\*[^*]+\*\*)"
    r"|(?<!\*)(\*[^*]+\*)(?!\*)"
    r"|(\[[^\]]+\]\([^)]+\))"
)


def shade(cell, hex_colour: str) -> None:
    """Apply a background fill to a table cell (python-docx has no API for this)."""
    element = OxmlElement("w:shd")
    element.set(qn("w:val"), "clear")
    element.set(qn("w:fill"), hex_colour)
    cell._tc.get_or_add_tcPr().append(element)


def add_hyperlink(paragraph, url: str, label: str) -> None:
    """Insert a real, clickable hyperlink: blue and underlined, so it reads as a link.

    python-docx has no hyperlink API, so the relationship and the w:hyperlink element are
    built by hand. Everything else in the document stays black; links are the one exception.
    """
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)

    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)

    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")

    colour = OxmlElement("w:color")
    colour.set(qn("w:val"), LINK_BLUE)
    props.append(colour)

    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(underline)

    run.append(props)

    text_element = OxmlElement("w:t")
    text_element.text = label
    run.append(text_element)

    link.append(run)
    # Appended in document order: preceding runs are already on the paragraph.
    paragraph._p.append(link)


def add_inline(paragraph, text: str) -> None:
    """Render inline markdown into runs on an existing paragraph."""
    pos = 0

    for match in INLINE.finditer(text):
        if match.start() > pos:
            paragraph.add_run(text[pos:match.start()])

        token = match.group(0)

        if token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = CODE_FONT
            run.font.size = Pt(9)
            run.font.color.rgb = BLACK
        elif token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
            run.font.color.rgb = BLACK
        elif token.startswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
            run.font.color.rgb = BLACK
        else:
            label, url = re.match(r"\[([^\]]+)\]\(([^)]+)\)", token).groups()
            add_hyperlink(paragraph, url, label)

        pos = match.end()

    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        run.font.color.rgb = BLACK


def add_code_block(doc: Document, lines: list[str]) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Pt(12)
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(8)

    run = paragraph.add_run("\n".join(lines))
    run.font.name = CODE_FONT
    run.font.size = Pt(8.5)
    run.font.color.rgb = BLACK


def is_separator(row: str) -> bool:
    """The |---|---| line under a table header."""
    return bool(re.fullmatch(r"\s*\|?[\s:\-|]+\|?\s*", row)) and "-" in row


def split_row(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in row.split("|")]


def add_table(doc: Document, rows: list[str]) -> None:
    header = split_row(rows[0])
    body = [split_row(r) for r in rows[2:]]
    width = max([len(header)] + [len(r) for r in body])

    table = doc.add_table(rows=1, cols=width)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    for i, cell in enumerate(table.rows[0].cells):
        shade(cell, HEADER_FILL)
        cell.text = ""
        paragraph = cell.paragraphs[0]
        add_inline(paragraph, header[i] if i < len(header) else "")
        for run in paragraph.runs:
            run.bold = True
            run.font.size = Pt(9)
            run.font.color.rgb = BLACK

    for record in body:
        cells = table.add_row().cells
        for i, cell in enumerate(cells):
            cell.text = ""
            paragraph = cell.paragraphs[0]
            add_inline(paragraph, record[i] if i < len(record) else "")
            for run in paragraph.runs:
                run.font.size = Pt(9)
                run.font.color.rgb = BLACK

    doc.add_paragraph()


def convert(md_path: Path, docx_path: Path) -> dict:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15

    # Word's built-in Heading and List styles carry a blue theme colour; override at the
    # style level so nothing inherits it even where runs are not set explicitly.
    for name in ["Title", "Heading 1", "Heading 2", "Heading 3", "Heading 4",
                 "List Bullet", "List Number"]:
        try:
            style = doc.styles[name]
        except KeyError:
            continue
        style.font.color.rgb = BLACK
        style.font.name = BODY_FONT
        if name.startswith("List"):
            style.font.size = Pt(10.5)

    stats = {"headings": 0, "tables": 0, "code_blocks": 0, "paragraphs": 0, "list_items": 0}
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # fenced code
        if stripped.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            add_code_block(doc, block)
            stats["code_blocks"] += 1
            continue

        # table
        if stripped.startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
            stats["tables"] += 1
            continue

        # heading - bold + underlined, black (Word's built-in Heading styles are blue)
        heading = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading:
            level = min(len(heading.group(1)), 4)
            paragraph = doc.add_heading("", level=level)
            add_inline(paragraph, heading.group(2))

            sizes = {1: 17, 2: 13.5, 3: 11.5, 4: 10.5}
            paragraph.paragraph_format.space_before = Pt(16 if level <= 2 else 12)
            paragraph.paragraph_format.space_after = Pt(6)

            for run in paragraph.runs:
                run.font.name = BODY_FONT
                run.font.size = Pt(sizes[level])
                run.font.color.rgb = BLACK
                run.bold = True
                run.underline = True

            stats["headings"] += 1
            i += 1
            continue

        # horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            rule = doc.add_paragraph()
            rule.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = rule.add_run("─" * 40)
            run.font.color.rgb = BLACK
            i += 1
            continue

        # blockquote
        if stripped.startswith(">"):
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Pt(18)
            add_inline(paragraph, " ".join(block))
            for run in paragraph.runs:
                run.italic = True
            stats["paragraphs"] += 1
            continue

        # lists
        bullet = re.match(r"^[-*]\s+(.*)", stripped)
        numbered = re.match(r"^(\d+)\.\s+(.*)", stripped)

        if bullet or numbered:
            style = "List Bullet" if bullet else "List Number"
            content = bullet.group(1) if bullet else numbered.group(2)
            paragraph = doc.add_paragraph(style=style)
            add_inline(paragraph, content)
            for run in paragraph.runs:
                run.font.color.rgb = BLACK
            stats["list_items"] += 1
            i += 1
            continue

        # paragraph: join soft-wrapped lines
        block = []
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#{1,6}\s|```|\||>|[-*]\s|\d+\.\s|-{3,}$)", lines[i].strip()
        ):
            block.append(lines[i].strip())
            i += 1

        if block:
            paragraph = doc.add_paragraph()
            add_inline(paragraph, " ".join(block))
            stats["paragraphs"] += 1

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(docx_path)

    return stats


def main():
    ap = argparse.ArgumentParser(description="Convert a Markdown report to .docx.")
    ap.add_argument("source")
    ap.add_argument("target")
    args = ap.parse_args()

    stats = convert(Path(args.source), Path(args.target))
    print(f"wrote {args.target}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
