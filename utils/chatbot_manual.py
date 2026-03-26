from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
import re
import textwrap
import unicodedata


@dataclass(frozen=True)
class ChatbotManualSection:
    title: str
    markdown: str
    source_path: str | None = None


def _project_root() -> Path:
    # utils/ -> project root
    return Path(__file__).resolve().parents[1]


def _read_markdown(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def get_chatbot_manual_sections() -> list[ChatbotManualSection]:
    root = _project_root()
    handout_path = root / "docs" / "IMNID_Chatbot_End_User_Handout.md"
    prompts_path = root / "docs" / "IMNID_Chatbot_CopyPaste_Prompts.md"

    sections: list[ChatbotManualSection] = []

    handout_md = _read_markdown(handout_path).strip()
    if handout_md:
        sections.append(
            ChatbotManualSection(
                title="End User Handout",
                markdown=handout_md + "\n",
                source_path=str(handout_path),
            )
        )

    prompts_md = _read_markdown(prompts_path).strip()
    if prompts_md:
        sections.append(
            ChatbotManualSection(
                title="Copy/Paste Prompts",
                markdown=prompts_md + "\n",
                source_path=str(prompts_path),
            )
        )

    if not sections:
        sections.append(
            ChatbotManualSection(
                title="Manual",
                markdown=(
                    "# IMNID Chatbot Manual\n\n"
                    "The manual files were not found in the `docs/` folder.\n"
                ),
                source_path=None,
            )
        )

    return sections


def build_chatbot_manual_markdown() -> str:
    sections = get_chatbot_manual_sections()
    return "\n\n---\n\n".join(section.markdown.strip() for section in sections).strip() + "\n"


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def generate_chatbot_manual_doc_bytes(markdown_text: str, *, title: str = "IMNID Chatbot Manual") -> bytes:
    """
    Generate a Word-readable .doc (HTML) file from markdown.

    - Uses the `markdown` package when available.
    - Falls back to a <pre> plain text HTML document if markdown is missing.
    """
    html_body = ""
    try:
        import markdown as md  # type: ignore

        html_body = md.markdown(
            markdown_text,
            extensions=["fenced_code", "tables", "sane_lists", "toc"],
            output_format="html5",
        )
    except Exception:
        html_body = f"<pre>{_html_escape(markdown_text)}</pre>"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{_html_escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      line-height: 1.35;
      font-size: 12pt;
      color: #111827;
      margin: 28px;
    }}
    h1, h2, h3 {{ margin: 0.9em 0 0.3em 0; }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 10.5pt;
    }}
    pre {{
      background: #f3f4f6;
      padding: 10px 12px;
      border-radius: 6px;
      overflow-x: auto;
    }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f9fafb; text-align: left; }}
    hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 16px 0; }}
    .meta {{ color: #6b7280; font-size: 10.5pt; margin-bottom: 14px; }}
  </style>
</head>
<body>
  <div class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
  {html_body}
</body>
</html>
"""
    return html.encode("utf-8")


def _pdf_safe(text: str) -> str:
    # ReportLab default fonts are not fully Unicode. Replace common punctuation, then strip remaining non-ASCII.
    replacements = {
        "—": "-",
        "–": "-",
        "‑": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "•": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def generate_chatbot_manual_pdf_bytes(markdown_text: str, *, title: str = "IMNID Chatbot Manual") -> bytes:
    """
    Generate a simple PDF from markdown text.

    This keeps formatting lightweight and avoids optional font dependencies.
    """
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.lib.units import inch  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore
    except Exception:
        return b""

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4

    margin_x = 0.75 * inch
    margin_top = 0.75 * inch
    margin_bottom = 0.75 * inch

    y = page_height - margin_top

    def new_page():
        nonlocal y
        pdf.showPage()
        y = page_height - margin_top

    def draw_wrapped(text: str, *, font_name: str, font_size: int, indent: float = 0.0, extra_gap: int = 0):
        nonlocal y
        pdf.setFont(font_name, font_size)
        max_width = page_width - (2 * margin_x) - indent
        approx_chars = max(24, int(max_width / (font_size * 0.55)))
        wrapped_lines = textwrap.wrap(_pdf_safe(text), width=approx_chars) or [""]
        for line in wrapped_lines:
            if y <= margin_bottom + font_size + 6:
                new_page()
                pdf.setFont(font_name, font_size)
            pdf.drawString(margin_x + indent, y, line)
            y -= font_size + 4
        if extra_gap:
            y -= extra_gap

    # Title
    draw_wrapped(title, font_name="Helvetica-Bold", font_size=16, extra_gap=6)
    draw_wrapped(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", font_name="Helvetica", font_size=10, extra_gap=8)

    in_code_block = False
    for raw in markdown_text.splitlines():
        line = raw.rstrip("\n")

        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        if not line.strip():
            y -= 6
            if y <= margin_bottom + 20:
                new_page()
            continue

        if in_code_block:
            draw_wrapped(line, font_name="Courier", font_size=9, indent=12.0)
            continue

        if line.startswith("# "):
            draw_wrapped(line[2:].strip(), font_name="Helvetica-Bold", font_size=14, extra_gap=4)
            continue
        if line.startswith("## "):
            draw_wrapped(line[3:].strip(), font_name="Helvetica-Bold", font_size=12, extra_gap=2)
            continue
        if line.startswith("### "):
            draw_wrapped(line[4:].strip(), font_name="Helvetica-Bold", font_size=11, extra_gap=2)
            continue

        bullet_match = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet_match:
            draw_wrapped(f"- {bullet_match.group(1)}", font_name="Helvetica", font_size=10, indent=10.0)
            continue

        draw_wrapped(line, font_name="Helvetica", font_size=10)

    pdf.save()
    return buffer.getvalue()

