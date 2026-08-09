#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.parsing.gsm8k_metrics import score_gsm8k_generations


def main():
    parser = argparse.ArgumentParser(description="Score a saved GSM8K result file")
    parser.add_argument("result_file", type=Path)
    args = parser.parse_args()

    with args.result_file.open(encoding="utf-8") as handle:
        results = json.load(handle)

    metrics = score_gsm8k_generations(results.get("generations", []))
    print(f"Result file: {args.result_file}")
    print(f"Correct: {metrics['correct']}/{metrics['evaluated']}")
    print(f"Parsed answers: {metrics['parsed_answers']}/{metrics['evaluated']}")
    print(f"Accuracy: {100 * metrics['accuracy']:.2f}%")
    print(f"Average NFE: {metrics['average_nfe']:.2f}")
    print(f"Total NFE: {metrics['total_nfe']:.0f}")
    print(f"NFE range: {metrics['min_nfe']:.0f}-{metrics['max_nfe']:.0f}")
    print(f"Average remasks: {metrics['average_remasks']:.2f}")
    print(f"Total remasks: {metrics['total_remasks']:.0f}")


if __name__ == "__main__":
    main()
