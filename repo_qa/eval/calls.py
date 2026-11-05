"""Measure how accurate the knowledge graph's call edges are, by hand-labelling a sample.

The graph builder writes every call decision to a CSV (--labeling-csv). Labelling thousands is
not realistic, so `sample_calls` draws a stratified random sample (some certain, some likely,
some guess edges) and shows the line of code where each call happens, which makes each label a
few seconds' work. `calls_precision` then turns the labelled sample into precision per
confidence level, with a confidence interval, because a small sample deserves an honest range.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from repo_qa.index.index import RepoIndex

LEVELS = ("certain", "likely", "guess")
SHARES = {"certain": 0.35, "likely": 0.25, "guess": 0.40}  # more weight on the uncertain edges
LABEL_COLUMN = "correct (y/n)"
_YES, _NO = {"y", "yes", "1", "true"}, {"n", "no", "0", "false"}


def read_calls_csv(path: str | Path) -> list[dict]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    needed = {"caller", "call_as_written", "resolved_to", "confidence_or_outcome", "via"}
    if rows and not needed <= set(rows[0]):
        raise ValueError(f"{path} does not look like a calls CSV (missing columns: {sorted(needed - set(rows[0]))}).")
    return rows


def call_site(index: RepoIndex, caller_id: str, call_name: str) -> str:
    """'path:line: code' for where the caller makes this call, or '' if it cannot be found."""
    if caller_id not in index.graph:
        return ""
    # Prefer the exact call as written ("mail.init_app("); fall back to the method name alone, which also
    # covers calls whose receiver the parser could not name ("?.to_dict").
    exact = None if call_name.startswith("?") else call_name + "("
    loose = call_name.rsplit(".", 1)[-1] + "("
    first = loose_hit = ""
    for chunk_id in index.graph.node(caller_id).get("chunk_ids", []):
        chunk = index.chunk(chunk_id)
        for offset, line in enumerate(chunk.text.split("\n")):
            site = f"{chunk.path}:{chunk.start_line + offset}: {line.strip()}"
            if exact and exact in line:
                return site
            if loose in line and not loose_hit:
                loose_hit = site
            first = first or site
    return loose_hit or first


def sample_calls(index: RepoIndex, rows: list[dict], n: int, seed: int = 1, *, include_external: bool = False) -> list[dict]:
    """A stratified, repeatable sample of resolved call edges, each with its call site.

    Calls into libraries (targets starting with "ext:") are left out by default: they test library
    detection, not the repository's own call graph, and would inflate the accuracy figure."""
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {level: [] for level in LEVELS}
    seen: set[tuple] = set()
    for row in rows:
        level, target = row["confidence_or_outcome"], row["resolved_to"]
        key = (row["caller"], row["call_as_written"], target)
        if level in pools and target and key not in seen and (include_external or not target.startswith("ext:")):
            seen.add(key)
            pools[level].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    quota = {level: min(len(pools[level]), round(n * SHARES[level])) for level in LEVELS}
    while sum(quota.values()) < n:  # hand the remainder to whichever levels still have rows
        spare = [lv for lv in LEVELS if quota[lv] < len(pools[lv])]
        if not spare:
            break
        quota[max(spare, key=lambda lv: len(pools[lv]) - quota[lv])] += 1
    chosen = [row for level in LEVELS for row in pools[level][: quota[level]]]
    rng.shuffle(chosen)
    return [
        {"id": i, "confidence": r["confidence_or_outcome"], "caller": r["caller"], "call_as_written": r["call_as_written"],
         "resolved_to": r["resolved_to"], "via": r["via"], "call_site": call_site(index, r["caller"], r["call_as_written"]),
         LABEL_COLUMN: ""}
        for i, r in enumerate(chosen, 1)
    ]


def write_sample(rows: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "confidence", "caller", "call_as_written", "resolved_to", "via", "call_site", LABEL_COLUMN]
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (reliable even for small samples)."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def calls_precision(path: str | Path) -> dict[str, dict]:
    """Precision per confidence level from a hand-labelled sample CSV."""
    counts = {level: {"correct": 0, "labelled": 0, "unlabelled": 0} for level in LEVELS}
    for row in read_calls_csv_labelled(path):
        level = row.get("confidence", "")
        if level not in counts:
            raise ValueError(f"Row {row.get('id')}: unknown confidence {level!r}.")
        label = (row.get(LABEL_COLUMN) or "").strip().lower()
        if not label:
            counts[level]["unlabelled"] += 1
        elif label in _YES | _NO:
            counts[level]["labelled"] += 1
            counts[level]["correct"] += label in _YES
        else:
            raise ValueError(f"Row {row.get('id')}: label must be y or n, got {label!r}.")
    result: dict[str, dict] = {}
    for level, c in counts.items():
        low, high = wilson_interval(c["correct"], c["labelled"])
        result[level] = {**c, "precision": c["correct"] / c["labelled"] if c["labelled"] else None,
                         "low": low, "high": high}
    return result


def read_calls_csv_labelled(path: str | Path) -> list[dict]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def precision_text(result: dict[str, dict]) -> str:
    lines = ["| Confidence | Labelled | Correct | Precision | 95% range |", "|---|---|---|---|---|"]
    for level in LEVELS:
        r = result[level]
        if r["labelled"]:
            lines.append(f"| {level} | {r['labelled']} | {r['correct']} | {r['precision'] * 100:.0f}% | "
                         f"{r['low'] * 100:.0f}% to {r['high'] * 100:.0f}% |")
        else:
            lines.append(f"| {level} | 0 | 0 | n/a | n/a |")
    pending = sum(r["unlabelled"] for r in result.values())
    if pending:
        lines.append(f"\n{pending} sampled edge(s) are still unlabelled and not counted.")
    return "\n".join(lines)
