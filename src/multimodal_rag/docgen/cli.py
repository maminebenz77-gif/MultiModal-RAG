"""Minimal CLI proving the human-in-the-loop escalation genuinely
survives a killed-and-restarted process -- not just a resume within
one still-running Python process, which an in-memory checkpointer
could do just as well.

Usage:
    First launch, from a state JSON file:
        uv run python -m multimodal_rag.docgen.cli --thread-id demo --state-file state.json

    If that run escalates, it prints the thread id and exits. Resume
    it later -- even after killing this process and starting a new
    one -- with just the thread id:
        uv run python -m multimodal_rag.docgen.cli --thread-id demo

This is deliberately not the full source-resolution/configuration
wizard described in the docgen build plan -- those are CLI glue for
other phases. This script exists to exercise checkpointer.py and
nodes/escalation.py end to end, for real, from the command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from langgraph.types import Command, RunnableConfig

from .checkpointer import build_checkpointer
from .graph import build_graph
from .nodes.escalation import HumanResponse
from .sources import SourceSpec
from .stack import build_stack

_ACTION_PROMPT = """
Choose one:
  [a] answer directly
  [r] reformulate -- give guidance, don't answer yourself
  [s] skip this question entirely
"""
_ACTION_KEYS: dict[str, Literal["answer", "reformulate", "skip"]] = {
    "a": "answer",
    "r": "reformulate",
    "s": "skip",
}


def _print_interrupt(interrupt: Any, thread_id: str) -> None:
    print("\nPAUSED -- a human needs to answer this question:")
    print(json.dumps(interrupt.value, indent=2, default=str))
    print(f"\nResume with: uv run python -m multimodal_rag.docgen.cli --thread-id {thread_id}")


def _prompt_for_response() -> HumanResponse:
    print(_ACTION_PROMPT)
    while True:
        choice = input("Action [a/r/s]: ").strip().lower()
        if choice in _ACTION_KEYS:
            break
        print(f"Not one of {sorted(_ACTION_KEYS)!r} -- try again.")
    action = _ACTION_KEYS[choice]
    text = "" if action == "skip" else input("Text: ")
    return {"action": action, "text": text}


def _print_final(result: dict[str, Any]) -> None:
    print("\nDONE:")
    print(json.dumps(result["answers"], indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument(
        "--state-file", type=Path, help="Initial DocGenState JSON -- only for a NEW run"
    )
    args = parser.parse_args()

    stack = build_stack()
    config: RunnableConfig = {"configurable": {"thread_id": args.thread_id}}

    with build_checkpointer() as checkpointer:
        graph = build_graph(stack, checkpointer=checkpointer)

        if args.state_file:
            initial_state = json.loads(args.state_file.read_text())
            # JSON has no concept of SourceSpec -- json.loads() hands
            # back plain dicts, but attempt_answer_node reads .role as
            # an attribute, not a key. Reconstruct the real dataclass
            # instances before this ever reaches the graph.
            initial_state["sources"] = [SourceSpec(**s) for s in initial_state["sources"]]
            result = graph.invoke(initial_state, config)
        else:
            snapshot = graph.get_state(config)
            if not snapshot.next:
                print(f"No run pending for thread {args.thread_id!r} -- pass --state-file.")
                return
            pending = snapshot.tasks[0].interrupts[0]
            print(json.dumps(pending.value, indent=2, default=str))
            response = _prompt_for_response()
            result = graph.invoke(Command(resume=response), config)

        if "__interrupt__" in result:
            _print_interrupt(result["__interrupt__"][0], args.thread_id)
        else:
            _print_final(result)


if __name__ == "__main__":
    main()
