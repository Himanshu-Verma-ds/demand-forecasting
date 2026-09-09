from __future__ import annotations

"""Replace a trailing section of a .docx with freshly rendered Markdown.

Used when a section already in the document needs to be regenerated without rebuilding the
whole file, so manual edits everywhere else survive. Finds the heading whose text contains
`--marker`, removes that block and everything after it, then renders the new Markdown in its
place.

    python scripts/replace_docx_section.py report.docx new_section.md --marker "benchmarked against"
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from docx import Document

from append_docx_section import render_into  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Replace a trailing .docx section from Markdown.")
    ap.add_argument("docx")
    ap.add_argument("section_md")
    ap.add_argument("--marker", required=True, help="Text identifying the heading to replace from")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    target = Path(args.docx)

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = target.with_name(f"{target.stem}.backup-{stamp}{target.suffix}")
        shutil.copy2(target, backup)
        print(f"backed up to {backup.name}")

    doc = Document(str(target))
    body = doc.element.body
    blocks = [c for c in body.iterchildren() if c.tag.endswith(("}p", "}tbl"))]

    start = None
    for i, block in enumerate(blocks):
        if args.marker in "".join(block.itertext()):
            start = i
            break

    if start is None:
        raise SystemExit(f"marker {args.marker!r} not found; nothing replaced")

    removed = 0
    for block in blocks[start:]:
        block.getparent().remove(block)
        removed += 1

    print(f"removed {removed} block(s) from index {start}")

    stats = render_into(doc, Path(args.section_md).read_text(encoding="utf-8").splitlines())
    doc.save(str(target))

    after = Document(str(target))
    print(f"rewrote section: paragraphs now {len(after.paragraphs)}, tables {len(after.tables)}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
