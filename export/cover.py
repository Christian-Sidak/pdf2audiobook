"""Embed cover art into finished audiobook files (M4B + chapter M4As).

Cover generation is out of pipeline (mflux / Qwen-Image); this just muxes a
PNG/JPEG into the container as attached_pic so Apple Books and every player
show it. Square art (1:1) is the audiobook convention.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console

from pipeline.config import OUTPUT_DIR

console = Console()

# Typefaces Qwen may art-direct with: name -> (regular ttc/idx, bold/display,
# italic). Classic book faces available on macOS.
FONT_FAMILIES = {
    "Didot":       (("/System/Library/Fonts/Supplemental/Didot.ttc", 0),
                    ("/System/Library/Fonts/Supplemental/Didot.ttc", 1),
                    ("/System/Library/Fonts/Supplemental/Didot.ttc", 2)),
    "Baskerville": (("/System/Library/Fonts/Supplemental/Baskerville.ttc", 0),
                    ("/System/Library/Fonts/Supplemental/Baskerville.ttc", 1),
                    ("/System/Library/Fonts/Supplemental/Baskerville.ttc", 2)),
    "Cochin":      (("/System/Library/Fonts/Supplemental/Cochin.ttc", 0),
                    ("/System/Library/Fonts/Supplemental/Cochin.ttc", 1),
                    ("/System/Library/Fonts/Supplemental/Cochin.ttc", 2)),
    "Hoefler Text":(("/System/Library/Fonts/Supplemental/Hoefler Text.ttc", 0),
                    ("/System/Library/Fonts/Supplemental/Hoefler Text.ttc", 1),
                    ("/System/Library/Fonts/Supplemental/Hoefler Text.ttc", 2)),
    "Palatino":    (("/System/Library/Fonts/Palatino.ttc", 0),
                    ("/System/Library/Fonts/Palatino.ttc", 1),
                    ("/System/Library/Fonts/Palatino.ttc", 3)),
    "Optima":      (("/System/Library/Fonts/Optima.ttc", 0),
                    ("/System/Library/Fonts/Optima.ttc", 1),
                    ("/System/Library/Fonts/Optima.ttc", 3)),
    "Georgia":     (("/System/Library/Fonts/Supplemental/Georgia.ttf", 0),
                    ("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 0),
                    ("/System/Library/Fonts/Supplemental/Georgia Italic.ttf", 0)),
}


def _font(family: str, weight: int, size: int):
    """weight: 0 regular, 1 display/bold, 2 italic."""
    from PIL import ImageFont
    fam = FONT_FAMILIES.get(family, FONT_FAMILIES["Baskerville"])
    path, idx = fam[weight]
    try:
        return ImageFont.truetype(path, size, index=idx)
    except Exception:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia.ttf", size)


def _letterspace(s: str, spaces: int = 1) -> str:
    return (" " * spaces).join(s)


def _hex(c, default):
    if isinstance(c, (list, tuple)) and len(c) == 3:
        return tuple(int(x) for x in c)
    if isinstance(c, str) and c.lstrip("#") and len(c.lstrip("#")) == 6:
        h = c.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return default


def compose_cover(art: str | Path, out: str | Path, title: str,
                  subtitle: str | None = None, author: str | None = None,
                  narrator: str | None = None, typography: dict | None = None) -> Path:
    """Composite title/author typography over generated art. Art stays
    textless; typography (font family + colors) is Qwen-art-directed to match
    the palette, falling back to a classic default."""
    from PIL import Image, ImageDraw

    typ = typography or {}
    family = typ.get("font", "Baskerville")
    if family not in FONT_FAMILIES:
        family = "Baskerville"
    title_c = _hex(typ.get("title_color"), (238, 232, 214))
    accent_c = _hex(typ.get("accent_color"), (212, 175, 90))
    scrim_c = _hex(typ.get("scrim_color"), (6, 10, 26))

    img = Image.open(art).convert("RGB")
    W, H = img.size

    # Scrim starts higher and reaches deeper so the title never fights the art
    # (e.g. gold letters over the gold key).
    scrim = Image.new("L", (1, H), 0)
    for y in range(H):
        t = max(0.0, (y - H * 0.42) / (H * 0.58))
        scrim.putpixel((0, y), int(240 * (t ** 1.25)))
    scrim = scrim.resize((W, H))
    img = Image.composite(Image.new("RGB", (W, H), scrim_c), img, scrim)

    draw = ImageDraw.Draw(img)

    def centered(text, weight, size, y, fill, ls=0, stroke=0):
        font = _font(family, weight, size)
        s = _letterspace(text, ls) if ls else text
        bb = draw.textbbox((0, 0), s, font=font)
        # Dark outline makes any color legible over any background.
        draw.text(((W - bb[2]) / 2, y), s, font=font, fill=fill,
                  stroke_width=stroke, stroke_fill=scrim_c)
        return bb[3]

    def _line_width(s, size):
        return draw.textbbox((0, 0), s, font=_font(family, 1, size))[2]

    # Fit the title: prefer wide letterspacing at 62pt, then trade spacing and
    # size down until the longest line fits inside the side margins. Long
    # period titles ('A New and Accurate Description of the...') overflow the
    # fixed layout otherwise.
    lines = _wrap(title.upper(), 18)
    max_w = int(W * 0.88)
    ls, size = 2, 62
    for try_ls in (2, 1, 0):
        for try_size in range(62, 30, -2):
            if all(_line_width(_letterspace(l, try_ls) if try_ls else l, try_size) <= max_w
                   for l in lines):
                ls, size = try_ls, try_size
                break
        else:
            continue
        break
    else:
        ls, size = 0, 32

    # Start high enough that the block never collides with the author line.
    block_h = int(len(lines) * size * 1.35)
    y = min(int(H * 0.60), int(H * 0.865) - block_h)
    for line in lines:
        y += int(centered(line, 1, size, y, title_c, ls=ls, stroke=3) * 1.1)
    if subtitle:
        y += 14
        for line in _wrap(subtitle, 40):
            y += int(centered(line, 2, 30, y, accent_c, stroke=2) * 1.15)
    if author:
        centered(author.upper(), 0, 34, int(H * 0.885), title_c, ls=3, stroke=2)
    if narrator:
        centered(f"Narrated by {narrator}", 2, 24, int(H * 0.935), accent_c, stroke=1)

    img.save(out, quality=95)
    return Path(out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def design_concept(book_text: str, title: str, author: str) -> dict:
    """Qwen reads the book and returns a FLUX/z-image prompt for abstract,
    textless cover art (+ palette/mood/negative)."""
    from pipeline.ollama_client import chat_json
    from pipeline.config import REWRITE_MODEL

    schema = {"type": "object", "required": ["prompt", "palette", "mood", "negative"],
              "properties": {k: {"type": "string"} for k in ("prompt", "palette", "mood", "negative")}}
    msg = [{"role": "user", "content": (
        f"Art director for audiobook covers. Book: {title} by {author}. Write an image-generation "
        "prompt for an ABSTRACT, elegant, TEXTLESS fine-art book jacket: one strong symbolic "
        "visual idea, atmospheric, no faces, no text, no literal stacks of books. Return JSON: "
        "prompt (~50 words vivid), palette, mood, negative. EXCERPT: " + book_text[:2000])}]
    return chat_json(REWRITE_MODEL, msg, schema, temperature=0.4)


def design_typography(concept: dict) -> dict:
    """Qwen picks a typeface (from installed families) and text colors that
    harmonize with the art's palette."""
    from pipeline.ollama_client import chat_json
    from pipeline.config import REWRITE_MODEL

    fonts = list(FONT_FAMILIES)
    schema = {"type": "object",
              "required": ["font", "title_color", "accent_color", "scrim_color", "reason"],
              "properties": {"font": {"type": "string", "enum": fonts},
                             **{k: {"type": "string"} for k in
                                ("title_color", "accent_color", "scrim_color", "reason")}}}
    msg = [{"role": "user", "content": (
        f"Book-cover typographer. Cover art: {concept.get('prompt','')}. "
        f"Palette: {concept.get('palette','')}. Mood: {concept.get('mood','')}. "
        f"Choose a typeface (one of: {', '.join(fonts)}) and text colors that harmonize "
        "elegantly for a serious literary audiobook. Return JSON: font, title_color (hex, "
        "legible over a dark scrim), accent_color (hex, palette accent), scrim_color (hex, deep "
        "base tint for the bottom gradient), reason (one line).")}]
    return chat_json(REWRITE_MODEL, msg, schema, temperature=0.3)


def _embed(audio: Path, image: Path) -> bool:
    tmp = audio.with_suffix(audio.suffix + ".tmp")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(audio), "-i", str(image),
           "-map", "0:a", "-map", "1:v", "-c:a", "copy", "-c:v", "mjpeg",
           "-disposition:v", "attached_pic", "-f", "mp4", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        console.print(f"[red]cover embed failed for {audio.name}:[/red] {r.stderr.strip()[:200]}")
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(audio)
    return True


def embed_cover(book: str, image: str) -> bool:
    """Embed `image` into a book's M4B and every chapter file under output/."""
    img = Path(image)
    if not img.exists():
        console.print(f"[red]cover image not found:[/red] {image}")
        return False
    book_dir = OUTPUT_DIR / book
    m4b = book_dir / "book.m4b"
    if not m4b.exists():
        console.print(f"[red]no audiobook at:[/red] {m4b}")
        return False

    targets = [m4b] + sorted((book_dir / "chapters").glob("*.m4a"))
    ok = 0
    for t in targets:
        if _embed(t, img):
            ok += 1
    # Keep the source art alongside the book for re-use / re-export.
    art_dest = book_dir / "cover.png"
    if img.resolve() != art_dest.resolve():
        import shutil
        shutil.copy(img, art_dest)
    console.print(f"[green]cover embedded into {ok}/{len(targets)} files[/green] "
                  f"[dim]({book_dir})[/dim]")
    return ok == len(targets)
