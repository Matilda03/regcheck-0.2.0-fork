"""Batch runner for clinical trial comparisons.

For each PDF in a folder, extracts the NCT registration ID from the paper text,
runs the clinical comparison via the RegCheck backend, and saves the result to
an output directory with a descriptive filename.

Usage:
    python batch_clinical.py --papers-dir ./papers --output-dir ./results

Full example:
    python batch_clinical.py \\
        --papers-dir ./papers \\
        --output-dir ./results \\
        --client openai \\
        --parser-choice grobid \\
        --output-format csv \\
        --reasoning-effort medium \\
        --dimensions-csv test_materials/dimensions_example.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF — already in requirements.txt

# ---------------------------------------------------------------------------
# Registration ID extraction
# ---------------------------------------------------------------------------

# Matches NCT followed by exactly 8 digits (ClinicalTrials.gov format).
_NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


def extract_nct_id_from_pdf(pdf_path: Path) -> str | None:
    """Return the first NCT ID found in the PDF text, or None if not found."""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logging.warning("Could not open %s: %s", pdf_path.name, exc)
        return None

    for page in doc:
        text = page.get_text()
        match = _NCT_PATTERN.search(text)
        if match:
            return match.group(0).upper()

    return None


# ---------------------------------------------------------------------------
# Output writing (mirrors backend/cli.py _write_output)
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "dimension",
    "paper_content_quotes",
    "paper_content_summary",
    "registration_content_quotes",
    "registration_content_summary",
    "deviation_judgement",
    "deviation_information",
]


def write_result(payload: dict, output_path: Path, output_format: str) -> None:
    if output_format == "json":
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    # CSV
    items = payload.get("items", []) if isinstance(payload, dict) else []
    rows: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append({k: "" if item.get(k) is None else str(item.get(k, "")) for k in _CSV_FIELDS})

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Per-paper comparison
# ---------------------------------------------------------------------------

async def compare_paper(
    pdf_path: Path,
    nct_id: str,
    output_path: Path,
    *,
    client: str,
    parser_choice: str,
    dimensions: list[dict] | None,
    append_previous_output: bool,
    reasoning_effort: str,
    output_format: str,
) -> None:
    """Run one clinical comparison and write the result file."""
    from backend.services.comparisons import clinical_trial_comparison

    result = await clinical_trial_comparison(
        nct_id,
        str(pdf_path),
        pdf_path.suffix.lower(),
        client,
        parser_choice=parser_choice,
        selected_dimensions=dimensions,
        append_previous_output=append_previous_output,
        reasoning_effort=reasoning_effort,
    )
    write_result(result.model_dump(), output_path, output_format)


# ---------------------------------------------------------------------------
# Dimensions CSV loader (mirrors backend/cli.py)
# ---------------------------------------------------------------------------

def load_dimensions(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        dims = []
        for row in reader:
            name = (
                row.get("dimension") or row.get("name") or
                row.get("Dimension") or row.get("Name") or ""
            ).strip()
            definition = (row.get("definition") or row.get("Definition") or "").strip()
            if name:
                dims.append({"dimension": name, "definition": definition})
    if not dims:
        raise ValueError(f"No dimensions found in {csv_path}")
    return dims


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def build_output_name(nct_id: str, pdf_path: Path, output_format: str) -> str:
    """Build a descriptive output filename: <NCT_ID>_<paper_stem>.<ext>"""
    ext = "json" if output_format == "json" else "csv"
    stem = re.sub(r"[^\w\-]", "_", pdf_path.stem)  # sanitise spaces / special chars
    return f"{nct_id}_{stem}.{ext}"


def run_batch(args: argparse.Namespace) -> None:
    papers_dir = Path(args.papers_dir)
    output_dir = Path(args.output_dir)

    if not papers_dir.is_dir():
        sys.exit(f"ERROR: --papers-dir does not exist or is not a directory: {papers_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    dimensions: list[dict] | None = None
    if args.dimensions_csv:
        dimensions_path = Path(args.dimensions_csv)
        if not dimensions_path.exists():
            sys.exit(f"ERROR: --dimensions-csv not found: {dimensions_path}")
        dimensions = load_dimensions(dimensions_path)
        logging.info("Loaded %d dimensions from %s", len(dimensions), dimensions_path)

    # Collect all PDFs (non-recursive; add rglob("*.pdf") if you want subdirectories)
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        sys.exit(f"No PDF files found in {papers_dir}")

    logging.info("Found %d PDF(s) in %s", len(pdf_files), papers_dir)

    successes: list[str] = []
    skipped: list[tuple[str, str]] = []   # (filename, reason)
    failures: list[tuple[str, str]] = []  # (filename, error)

    for pdf_path in pdf_files:
        logging.info("--- Processing: %s", pdf_path.name)

        # 1. Extract NCT ID
        nct_id = extract_nct_id_from_pdf(pdf_path)
        if nct_id is None:
            reason = "No NCT ID found in PDF text"
            logging.warning("SKIP %s — %s", pdf_path.name, reason)
            skipped.append((pdf_path.name, reason))
            continue

        logging.info("Found registration ID: %s", nct_id)

        # 2. Determine output path
        out_name = build_output_name(nct_id, pdf_path, args.output_format)
        output_path = output_dir / out_name

        if output_path.exists() and not args.overwrite:
            reason = f"Output already exists: {output_path.name} (use --overwrite to replace)"
            logging.warning("SKIP %s — %s", pdf_path.name, reason)
            skipped.append((pdf_path.name, reason))
            continue

        # 3. Run comparison
        try:
            asyncio.run(
                compare_paper(
                    pdf_path,
                    nct_id,
                    output_path,
                    client=args.client,
                    parser_choice=args.parser_choice,
                    dimensions=dimensions,
                    append_previous_output=args.append_previous_output,
                    reasoning_effort=args.reasoning_effort,
                    output_format=args.output_format,
                )
            )
            logging.info("Saved result → %s", output_path)
            successes.append(f"{pdf_path.name} ({nct_id}) → {output_path.name}")
        except Exception as exc:
            logging.error("FAILED %s — %s", pdf_path.name, exc, exc_info=True)
            failures.append((pdf_path.name, str(exc)))

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE — {len(pdf_files)} PDF(s) processed")
    print("=" * 60)

    print(f"\n  Succeeded : {len(successes)}")
    for item in successes:
        print(f"    ✓  {item}")

    if skipped:
        print(f"\n  Skipped   : {len(skipped)}")
        for name, reason in skipped:
            print(f"    –  {name}: {reason}")

    if failures:
        print(f"\n  Failed    : {len(failures)}")
        for name, err in failures:
            print(f"    ✗  {name}: {err}")

    print()

    if failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run RegCheck clinical comparisons for all PDFs in a folder. "
            "Each PDF must contain an NCT registration ID in its text."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--papers-dir",
        required=True,
        metavar="DIR",
        help="Folder containing the paper PDFs to process.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Folder where result files will be written.",
    )

    # LLM / parser options (match CLI)
    parser.add_argument(
        "--client",
        default="openai",
        choices=["openai", "deepseek", "groq"],
        help="LLM provider to use. (default: openai)",
    )
    parser.add_argument(
        "--parser-choice",
        default="grobid",
        choices=["grobid", "dpt2"],
        help="PDF parser for extracting paper text. (default: grobid)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort for OpenAI models. (default: medium)",
    )
    parser.add_argument(
        "--append-previous-output",
        action="store_true",
        help="Pass prior dimension responses into later prompts.",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "json"],
        help="Output format for each result file. (default: csv)",
    )

    # Optional dimensions override
    parser.add_argument(
        "--dimensions-csv",
        metavar="FILE",
        help="Optional CSV with 'dimension'/'definition' columns to override defaults.",
    )

    # Batch behaviour
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite output files that already exist.",
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_batch(build_parser().parse_args())
