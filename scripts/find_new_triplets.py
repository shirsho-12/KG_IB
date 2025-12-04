#!/usr/bin/env python3
"""
Utility to inspect clustering artifacts and list new triplets that were not
present in the original extraction output.

Example:
    python scripts/find_new_triplets.py \
        --artifacts output/rel_clustering/artifacts.json \
        --baseline output/webnlg/triplets.txt
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple, Literal, Optional

Triple = Tuple[str, str, str]
BaselineFormat = Literal["tuple_lines", "json_array_lines"]


def _parse_json_line(line: str) -> Optional[List[Triple]]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, list):
        return None

    triples: List[Triple] = []
    for item in payload:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(part, str) for part in item)
        ):
            return None
        triples.append((item[0], item[1], item[2]))
    return triples


def load_baseline_triplets(path: Path) -> Tuple[Set[Triple], BaselineFormat]:
    """
    Load baseline triplets from a file. Each line should contain a tuple literal,
    e.g. ("Alice", "knows", "Bob").
    """
    triplets: Set[Triple] = set()
    baseline_format: BaselineFormat = "tuple_lines"

    if not path.exists():
        return triplets, baseline_format

    lines = path.read_text().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        json_triplets = _parse_json_line(line)
        if json_triplets is not None:
            baseline_format = "json_array_lines"
            for triplet in json_triplets:
                triplets.add(triplet)
        else:
            triplet = ast.literal_eval(line)
            triplets.add((triplet[0], triplet[1], triplet[2]))
    return triplets, baseline_format


def load_artifact_triplets(path: Path) -> List[Triple]:
    """
    Read the JSON artifact (either rel_clustering or reg_clust) and extract all
    triplets that were processed by the clusterer.
    """
    data = json.loads(path.read_text())
    triplets: List[Triple] = []
    for fact in data.get("facts", []):
        triplets.append((fact["head"], fact["relation"], fact["tail"]))
    return triplets


def write_output(
    triplets: Sequence[Triple],
    output_path: Path,
    baseline_format: BaselineFormat,
) -> None:
    """
    Write new triplets to disk (JSON list of [head, relation, tail]).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [[h, r, t] for (h, r, t) in triplets]
    indent = None if baseline_format == "json_array_lines" else 2
    text = json.dumps(payload, indent=indent)
    if not text.endswith("\n"):
        text += "\n"
    output_path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find triplets present in clustering artifacts but absent "
        "from the original extraction file."
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("output/rel_clustering/artifacts.json"),
        help="Path to artifacts.json produced by rel_clustering or reg_clust.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("output/webnlg/triplets.txt"),
        help="Baseline triplets file; defaults to output/webnlg/triplets.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the list of new triplets (JSON).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.artifacts.exists():
        raise FileNotFoundError(f"Artifacts file not found: {args.artifacts}")

    baseline_triplets, baseline_format = load_baseline_triplets(args.baseline)
    artifact_triplets = load_artifact_triplets(args.artifacts)

    new_triplets = [
        triple for triple in artifact_triplets if triple not in baseline_triplets
    ]

    print(f"Loaded {len(artifact_triplets)} triplets from artifacts.")
    if baseline_triplets:
        print(f"Baseline contains {len(baseline_triplets)} triplets.")
    else:
        print("Baseline file missing or empty; treating all artifact triplets as new.")

    if new_triplets:
        print(f"Identified {len(new_triplets)} new triplets:\n")
        for head, relation, tail in new_triplets:
            print(f"({head}, {relation}, {tail})")
    else:
        print("No new triplets found.")

    if args.output:
        write_output(new_triplets, args.output, baseline_format)
        print(f"\nWrote new triplets to {args.output}")


if __name__ == "__main__":
    main()
