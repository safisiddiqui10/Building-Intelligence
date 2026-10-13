"""Turn the supplied BNS and IPC PDFs into section-level JSONL files.

Run: ``uv run python -m building_with_rag.ingestion``

Parsing only. No database, embeddings or network access. Text wording is kept as
printed; only page furniture, extra spaces and line-break hyphenation are cleaned.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pypdf
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
PARSER = "pypdf"
PARSER_VERSION = pypdf.__version__
EXTRACTOR_VERSION = "1"  # bump when the extraction rules change

REQUIRED_FIELDS = (
    "section_id", "act", "act_label", "status", "chapter", "chapter_title",
    "section_number", "heading", "text", "source_pdf", "source_sha256", "parser",
    "parser_version", "source_status_version", "needs_review",
)

ACTS = {
    "bns": {
        "act": "BNS_2023", "status": "in_force", "pdf": "BNS_2023_bare_act.pdf",
        "out": "bns_sections.jsonl", "title": r"BHARATIYA NYAYA SANHITA, \d{4}",
    },
    "ipc": {
        "act": "IPC_1860", "status": "repealed", "pdf": "IPC_1860_bare_act.pdf",
        "out": "ipc_sections.jsonl", "title": r"THE INDIAN PENAL CODE, \d{4}",
    },
}

CHAPTER_RE = re.compile(r"^(?:\d+\*\[)?CHAPTER\s+([IVXLC]+A?)\s*$")
FURNITURE_RE = re.compile(r"HomePage|^Note: For Reference only|^[-—_]{5,}$|^\d{1,3}$")
SEP_RE = re.compile(r"-\s*-|(?<=[.\]])-(?=[A-Za-z(\"“]|$)")  # heading/body separator: "--", "- -", ".-"
FOOTNOTE_RE = re.compile(r"^(Subs|Ins|Rep|Added|Omitted|The |S\. |Sec|Cl|Re-?num)")


@dataclass
class Line:
    text: str
    page: int
    chapter: str | None = None
    chapter_title: str | None = None


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_upper(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def read_pages(pdf: Path) -> list[str]:
    return [(p.extract_text() or "") for p in PdfReader(str(pdf)).pages]


def to_lines(pages: list[str], first_page: int = 1, last_page: int | None = None) -> list[Line]:
    """Clean lines, drop page furniture, and tag each line with its chapter."""
    out: list[Line] = []
    chapter = title = None
    for pno, text in enumerate(pages, start=1):
        if pno < first_page or (last_page and pno > last_page):
            continue
        raw = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
        raw = [ln for ln in raw if ln and not FURNITURE_RE.search(ln)]
        i = 0
        while i < len(raw):
            m = CHAPTER_RE.match(raw[i])
            if m:
                chapter, parts = m.group(1), []
                i += 1
                while i < len(raw) and is_upper(raw[i]) and not CHAPTER_RE.match(raw[i]):
                    parts.append(raw[i])
                    i += 1
                if parts:
                    title = " ".join(parts)
                continue
            out.append(Line(raw[i], pno, chapter, title))
            i += 1
    return out


def clean_text(lines: list[str]) -> str:
    """Join lines, repairing only line-break hyphenation."""
    out: list[str] = []
    for ln in lines:
        if out and re.search(r"[A-Za-z]-$", out[-1]) and ln[:1].islower():
            out[-1] += ln
        else:
            out.append(ln)
    return "\n".join(out).strip()


@dataclass
class Found:
    number: int
    label: str
    start: int  # index of the line holding the section number
    cut: int  # index where this section's own heading/margin block begins
    heading: str
    body_first: str  # first line text with number (and heading) removed
    review: list[str]


# ---------------------------------------------------------------- BNS


def bns_index(pages: list[str], body_page: int) -> tuple[dict[int, str], set[str]]:
    """Headings from the arrangement of sections: {number: heading}, plus group headings."""
    stop = next(
        (i for i, t in enumerate(pages, 1) if "CORRESPONDING SECTION TABLE" in t.upper() and i > 1),
        body_page,
    )
    lines = [ln.text for ln in to_lines(pages, 2, stop - 1)]
    headings: dict[int, str] = {}
    groups: set[str] = set()
    expected, cur, parts = 1, None, []
    pending: list[str] = []

    def close():
        nonlocal cur, parts
        if cur is not None:
            headings[cur] = " ".join(parts).strip()
        cur, parts = None, []

    for ln in lines:
        if ln in ("SECTION", "ARRANGEMENT OF SECTIONS") or ln.startswith("Index"):
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", ln)
        if m and int(m.group(1)) == expected:
            close()
            cur, parts = expected, [m.group(2)]
            expected += 1
        elif cur is not None and not parts[-1].rstrip().endswith("."):
            parts.append(ln)
        else:
            close()
            if not is_upper(ln):
                pending.append(ln)
                groups.add(norm(" ".join(pending[-2:])))
                groups.add(norm(ln))
    close()
    return headings, groups


def find_bns(lines: list[Line], headings: dict[int, str], groups: set[str]) -> list[Found]:
    found: list[Found] = []
    expected = 1
    for idx, ln in enumerate(lines):
        m = re.match(r"^(\d+)\.\s*\S", ln.text)
        if not (m and int(m.group(1)) == expected):
            continue
        n = expected
        expected += 1
        review: list[str] = []
        # heading = the lines directly above, matching the arrangement-of-sections heading
        want = norm(headings.get(n, ""))
        k_found = 0
        for k in range(1, 6):
            if idx - k < 0:
                break
            if norm("".join(x.text for x in lines[idx - k : idx])) == want and want:
                k_found = k
                break
        if not k_found and want:  # wording differs slightly from the index: accept a close match
            scored = [
                (SequenceMatcher(None, norm("".join(x.text for x in lines[idx - k : idx])), want).ratio(), k)
                for k in range(1, 6) if idx - k >= 0
            ]
            best = max(scored, default=(0, 0))
            if best[0] >= 0.9:
                k_found, review = best[1], ["heading wording differs from arrangement of sections"]
        if k_found:
            head = " ".join(x.text for x in lines[idx - k_found : idx])
        else:
            head, review = headings.get(n, ""), ["heading not matched to body"]
        cut = idx - k_found
        for k in (3, 2, 1):  # group heading such as "Of abetment" above the heading
            if cut - k >= 0 and norm("".join(x.text for x in lines[cut - k : cut])) in groups:
                cut -= k
                break
        body_first = re.sub(r"^\d+\.\s*", "", ln.text)
        found.append(Found(n, str(n), idx, cut, head.strip(), body_first, review))
    return found


# ---------------------------------------------------------------- IPC


def in_sequence(n: int, suf: str, last_n: int, last_suf: str) -> bool:
    return (n == last_n and suf > last_suf) or (last_n < n <= last_n + 15 and suf == "")


def find_ipc(lines: list[Line]) -> list[Found]:
    found: list[Found] = []
    last_n, last_suf = 0, ""
    for idx, ln in enumerate(lines):
        m = re.match(r"^(?:\d+\*\[)?(\d+)([A-Z]{0,2})(?:\. | (?=[\"“]))(\S.*)$", ln.text)
        if not m:
            continue
        n, suf = int(m.group(1)), m.group(2)
        mismatch = False
        if not in_sequence(n, suf, last_n, last_suf):
            # the body line may carry a misprinted number; trust the margin note if it fits
            margin = next(
                (mm for k in range(idx - 1, max(0, idx - 6) - 1, -1)
                 if (mm := re.match(r"^(?:\d+\*\[)?(\d+)([A-Z]{0,2})\.$", lines[k].text))),
                None,
            )
            if not (margin and int(margin.group(1)) == last_n + 1 and not margin.group(2)):
                continue
            n, suf, mismatch = int(margin.group(1)), margin.group(2), True
        head_parts, body_first = [], None
        for j in range(3):
            if idx + j >= len(lines):
                break
            t = lines[idx + j].text
            sep = SEP_RE.search(t)
            if sep:
                head_parts.append(t[: sep.start()])
                body_first = t[sep.end() :]
                break
            head_parts.append(t)
        if body_first is None:
            continue
        head = re.sub(r"^(?:\d+\*\[)?\d+[A-Z]{0,2}\.? ", "", " ".join(head_parts)).strip()
        label = f"{n}{suf}"
        review: list[str] = ["number in body differs from margin note"] if mismatch else []
        # the PDF repeats each section's number and heading as a margin note just above it
        cut = idx
        for m_idx in range(max(0, idx - 6), idx):
            if re.sub(r"^\d+\*\[", "", lines[m_idx].text).strip() == f"{label}.":
                cut = m_idx
                break
        if cut != idx:  # a group heading such as "Of fraudulent deeds ..." may sit above the margin note
            for k in (3, 2, 1):
                grp = [x.text for x in lines[cut - k : cut]] if cut - k >= 0 else []
                if grp and grp[0].startswith("Of ") and not re.search(r"[.;:,—]$", grp[-1]):
                    cut -= k
                    break
        if cut == idx:
            if FOOTNOTE_RE.match(head) or n > last_n + 1:
                continue  # a footnote or stray line, not a section
            review.append("margin note not found")
        found.append(Found(n, label, idx, cut, head, body_first.strip(), review))
        last_n, last_suf = n, suf
    return found


# ---------------------------------------------------------------- build


def build_records(key: str, pdf: Path, digest: str, status_version: str | None, log: list[str]):
    cfg = ACTS[key]
    pages = read_pages(pdf)
    empty_pages = [i for i, t in enumerate(pages, 1) if not t.strip()]
    title_m = re.search(cfg["title"], pages[0])
    act_label = title_m.group(0).title() if title_m else None

    if key == "bns":
        body_page = next(
            (i for i, t in enumerate(pages, 1) if re.search(r"^1\. \(1\) This Act may be called", t, re.MULTILINE)),
            None,
        )
        if body_page is None:
            raise RuntimeError("BNS body start (section 1) not found")
        headings, groups = bns_index(pages, body_page)
        log.append(f"bns: {len(headings)} headings read from the arrangement of sections")
        lines = to_lines(pages, body_page)
        found = find_bns(lines, headings, groups)
    else:
        lines = to_lines(pages)
        found = find_ipc(lines)

    records = []
    for i, f in enumerate(found):
        end = found[i + 1].cut if i + 1 < len(found) else len(lines)
        body = [f.body_first] + [x.text for x in lines[f.start + 1 : end]]
        if key == "ipc":
            # IPC body_first may sit on a later line than f.start when the heading wraps
            skip = 0
            for j in range(3):
                skip = j
                if SEP_RE.search(lines[f.start + j].text):
                    break
            body = [f.body_first] + [x.text for x in lines[f.start + skip + 1 : end]]
        text = clean_text([b for b in body if b])
        review = list(f.review)
        chapter, ctitle = lines[f.start].chapter, lines[f.start].chapter_title
        if not chapter or not ctitle:
            review.append("chapter not read")
        if not f.heading:
            review.append("heading missing")
        if len(text) < 20:
            review.append("very short text")
        if status_version is None or act_label is None:
            review.append("provenance value unknown")
        p0, p1 = lines[f.start].page, lines[max(f.start, end - 1)].page
        if any(p0 <= p <= p1 for p in empty_pages):
            review.append("page without extractable text (scanned?)")
        if not text:
            log.append(f"{key}: section {f.label} has no text; left out")
            continue
        records.append({
            "section_id": f"{key}:{f.number}" if not f.label[len(str(f.number)):] else f"{key}:{f.label}",
            "act": cfg["act"],
            "act_label": act_label,
            "status": cfg["status"],
            "chapter": chapter,
            "chapter_title": ctitle,
            "section_number": f.number,
            "heading": f.heading,
            "text": text,
            "source_pdf": cfg["pdf"],
            "source_sha256": digest,
            "parser": PARSER,
            "parser_version": PARSER_VERSION,
            "source_status_version": status_version,
            "needs_review": bool(review),
            "section_label": f.label,
            "page_start": p0,
            "page_end": p1,
            "extractor_version": EXTRACTOR_VERSION,
            "review_reasons": review,
        })
    if empty_pages:
        log.append(f"{key}: pages without extractable text: {empty_pages}")
    nums = sorted({r["section_number"] for r in records})
    gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in nums]
    if gaps:
        log.append(f"{key}: section numbers not found in the PDF body (repealed or omitted?): {gaps}")
    ids = [r["section_id"] for r in records]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise RuntimeError(f"{key}: duplicate section ids {sorted(dupes)}")
    return records


# ---------------------------------------------------------------- provenance / output


def read_provenance() -> dict[str, dict[str, str | None]]:
    """Per-PDF hash and status text, taken only from PROVENANCE.md."""
    path = RAW_DIR / "PROVENANCE.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    flat = re.sub(r"\s+", " ", text)
    info: dict[str, dict[str, str | None]] = {}
    for key, cfg in ACTS.items():
        row = re.search(re.escape(cfg["pdf"]) + r"`.*?\| (\d{4}-\d{2}-\d{2}) \| `([0-9a-f]{64})`", flat)
        note = re.search(re.escape(cfg["pdf"]) + r"` is .*?(\*\*in force\*\*|\*\*repealed\*\* as of [0-9]+ \w+ [0-9]{4})", flat)
        version = None
        if row and note:
            version = f"PROVENANCE.md: retrieved {row.group(1)}; {note.group(1).replace('**', '')}"
        info[key] = {"sha256": row.group(2) if row else None, "version": version}
    return info


def existing_is_current(path: Path, digest: str) -> bool:
    if not path.exists():
        return False
    try:
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    except ValueError:
        return False
    return bool(rows) and all(
        r.get("source_sha256") == digest
        and r.get("parser_version") == PARSER_VERSION
        and r.get("extractor_version") == EXTRACTOR_VERSION
        and all(f in r for f in REQUIRED_FIELDS)
        for r in rows
    )


def write_atomic(path: Path, records: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> int:
    prov = read_provenance()
    status = 0
    for key, cfg in ACTS.items():
        pdf, out = RAW_DIR / cfg["pdf"], OUT_DIR / cfg["out"]
        if not pdf.exists():
            print(f"{key}: MISSING input {pdf.relative_to(ROOT)}; output not touched")
            status = 1
            continue
        digest = sha256(pdf)
        if prov[key]["sha256"] != digest:
            print(f"{key}: hash differs from PROVENANCE.md (or not listed); treated as changed source")
        if existing_is_current(out, digest):
            print(f"{key}: {out.relative_to(ROOT)} already current; left unchanged")
            continue
        log: list[str] = []
        records = build_records(key, pdf, digest, prov[key]["version"], log)
        write_atomic(out, records)
        flagged = sum(r["needs_review"] for r in records)
        print(f"{key}: wrote {len(records)} records ({flagged} needs_review) to {out.relative_to(ROOT)}")
        for line in log:
            print("  " + line)
    return status


if __name__ == "__main__":
    sys.exit(main())
