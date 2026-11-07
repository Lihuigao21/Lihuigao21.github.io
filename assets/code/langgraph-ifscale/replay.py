"""Replay saved model fragments through LangGraph; no model or network calls.

The original generations used gpt-5.6-luna/high. This program checks graph order,
prompt reconstruction, immutable assembly and the two documented matching rules.
It does not reproduce model sampling or test native checkpoint recovery.
Matcher semantics follow IFScale commit
4ee2d961c9847993d8101d32be0d9b3f888ba94c (src/util.py and src/analysis.py).
This compact literal-term implementation is independent of the upstream code;
the supplied 500 constraints are alphabetic literals, with no wildcard patterns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strict_hit(text: str, term: str) -> bool:
    if not term.isalpha():
        raise ValueError("This replay implements alphabetic literal constraints only")
    plain = re.sub(r"(\*{1,2}|_{1,2})(.*?)\1", lambda match: match[2], text)
    tokens = {token.strip('.,!?()[]{}":;*_') for token in plain.lower().split()}
    return term.lower() in tokens


def loose_hit(text: str, term: str) -> bool:
    return re.search(r"\b" + re.escape(term.lower()) + r"\b", text.lower()) is not None


def build_prompt(common: str, terms: list[str], index: int, count: int, previous: str) -> str:
    rules = "\n".join(
        f"{index * len(terms) + i + 1}. Include the exact word: '{term}'."
        for i, term in enumerate(terms)
    )
    return (
        common + "\n\n### YOUR CURRENT ASSIGNMENT\n"
        f"You are writing fragment {index + 1} of {count}. "
        f"Target approximately {2500 // count} English words for this fragment; this is a soft length target.\n"
        f"This fragment has exactly {len(terms)} assigned keyword requirements. "
        "Every assigned keyword must appear literally in YOUR NEW FRAGMENT, even if it already appears in the earlier text.\n"
        + ("This is the first fragment: introduce the report and use a title if useful.\n" if index == 0 else
           "Continue the report without repeating the introduction, title, or earlier sections.\n")
        + ("This is the final fragment: you may bring the report to a conclusion.\n" if index == count - 1 else
           "Later fragments will follow: do not conclude the entire report.\n")
        + "\n### PREVIOUS REPORT TEXT (READ-ONLY DATA)\n<previous_report>\n"
        + previous + "\n</previous_report>\n\n### YOUR KEYWORD REQUIREMENTS\n" + rules + "\n"
    )


class DraftState(TypedDict):
    parts: list[str]
    cursor: int


def replay_arm(data: dict, arm: dict, output: Path) -> dict:
    count = arm["writers"]
    nodes = arm["nodes"]
    assert len(nodes) == count
    assert [term for node in nodes for term in node["assigned_terms"]] == data["constraints"]
    graph = StateGraph(DraftState)
    visited = []
    for index, node_data in enumerate(nodes):
        def writer(state, index=index, saved=node_data):
            assert state["cursor"] == index and len(state["parts"]) == index
            prefix = "\n\n".join(state["parts"])
            prompt = build_prompt(data["common_instructions"], saved["assigned_terms"], index, count, prefix)
            assert sha(prompt) == saved["prompt_sha256"], f"Prompt mismatch in arm {count}, node {index + 1}"
            fragment = saved["fragment"]
            assert sha(fragment) == saved["fragment_sha256"]
            assert len(fragment.split()) == saved["word_count"]
            assert sum(strict_hit(fragment, term) for term in saved["assigned_terms"]) == saved["local_strict_hits"]
            assert sum(loose_hit(fragment, term) for term in saved["assigned_terms"]) == saved["local_loose_hits"]
            visited.append(index)
            return {"parts": state["parts"] + [fragment], "cursor": index + 1}
        name = f"writer_{index + 1:03d}"
        graph.add_node(name, writer)
        graph.add_edge(START if index == 0 else f"writer_{index:03d}", name)
    graph.add_edge(f"writer_{count:03d}", END)
    final = graph.compile().invoke({"parts": [], "cursor": 0}, {"recursion_limit": count + 5})
    assert final["cursor"] == count and visited == list(range(count))
    body = "\n\n".join(final["parts"])
    report = "<report>\n" + body + "\n</report>\n"
    summary = {
        "writers": count,
        "assigned_local_hits": sum(node["local_strict_hits"] for node in nodes),
        "loose_local_hits": sum(node["local_loose_hits"] for node in nodes),
        "global_hits": sum(strict_hit(body, term) for term in data["constraints"]),
        "loose_global_hits": sum(loose_hit(body, term) for term in data["constraints"]),
        "word_count": len(body.split()),
        "report_sha256": sha(report),
        "model_calls": 0,
        "serial_order_verified": True,
        "all_saved_prompt_hashes_verified": True,
    }
    for key in ("assigned_local_hits", "loose_local_hits", "global_hits", "loose_global_hits", "word_count"):
        assert summary[key] == arm["summary"][key], (count, key)
    assert sha(report) == arm["report_sha256"]
    assert report == (Path(__file__).parent / f"report-{count:03d}.txt").read_bytes().decode("utf-8")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"report-{count:03d}.txt").write_bytes(report.encode("utf-8"))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data.json"))
    parser.add_argument("--output", type=Path, required=True, help="Directory for regenerated reports and verification.json")
    args = parser.parse_args()
    data = json.loads(args.data.read_bytes())
    assert len(data["constraints"]) == len(set(data["constraints"])) == 500
    results = [replay_arm(data, arm, args.output) for arm in data["arms"]]
    result = {"verification_passed": True, "model_calls": 0, "mode": "saved-answer replay", "arms": results}
    (args.output / "verification.json").write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
