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
from .nodes.answering import DEFAULT_MAX_RETRIES
from .nodes.configuration import ConfirmationResponse
from .nodes.escalation import HumanResponse
from .sources import SourceSpec
from .stack import build_stack

_ESCALATION_PROMPT = """
Choose one:
  [a] answer directly
  [r] reformulate -- give guidance, don't answer yourself
  [s] skip this question entirely
"""
_ESCALATION_KEYS: dict[str, Literal["answer", "reformulate", "skip"]] = {
    "a": "answer",
    "r": "reformulate",
    "s": "skip",
}


def _print_interrupt(interrupt: Any, thread_id: str) -> None:
    print("\nPAUSED -- a human needs to respond:")
    print(json.dumps(interrupt.value, indent=2, default=str))
    print(f"\nResume with: uv run python -m multimodal_rag.docgen.cli --thread-id {thread_id}")


def _prompt_for_escalation_response() -> HumanResponse:
    print(_ESCALATION_PROMPT)
    while True:
        choice = input("Action [a/r/s]: ").strip().lower()
        if choice in _ESCALATION_KEYS:
            break
        print(f"Not one of {sorted(_ESCALATION_KEYS)!r} -- try again.")
    action = _ESCALATION_KEYS[choice]
    text = "" if action == "skip" else input("Text: ")
    return {"action": action, "text": text}


def _prompt_for_confirmation_response() -> ConfirmationResponse:
    choice = input("\nConfirm this plan? [y/n]: ").strip().lower()
    if choice in ("y", "yes"):
        return {"action": "confirm", "text": ""}
    text = input("What should change? ")
    return {"action": "revise", "text": text}


def _prompt_for_pending_response(pending: Any) -> HumanResponse | ConfirmationResponse:
    """Two different pauses exist in this graph, with two different
    expected resume shapes -- reformulate_for_confirmation's payload
    always has a "summary" key, ask_human's never does, so that's
    enough to tell them apart without the caller needing to track
    which node is paused."""
    if "summary" in pending.value:
        return _prompt_for_confirmation_response()
    return _prompt_for_escalation_response()


def _print_final(result: dict[str, Any]) -> None:
    print("\nDONE:")
    print(json.dumps(result["answers"], indent=2, default=str))
    print(f"Total LLM calls: {result['usage']['llm_calls']}")


def _run_streaming(graph: Any, run_input: Any, config: RunnableConfig) -> dict[str, Any]:
    """Drives the graph via stream() instead of invoke() so the running
    LLM-call count can be shown live, as it actually happens, instead
    of only being knowable once the whole run (or the run up to the
    next pause) has already finished. The LAST yielded chunk is the
    exact same final/paused state invoke() would have returned."""
    last_count = 0
    chunk: dict[str, Any] = {}
    for chunk in graph.stream(run_input, config, stream_mode="values"):
        count = chunk.get("usage", {}).get("llm_calls", last_count)
        if count != last_count:
            print(f"  ...LLM calls so far: {count}")
            last_count = count
    return chunk


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
            # A hand-written state file shouldn't need to know about
            # internal bookkeeping fields -- backfill sensible defaults
            # for anything it left out, rather than a confusing KeyError
            # deep inside the graph.
            initial_state["configuration"].setdefault("max_retries", DEFAULT_MAX_RETRIES)
            initial_state.setdefault("usage", {"llm_calls": 0})
            result = _run_streaming(graph, initial_state, config)
        else:
            snapshot = graph.get_state(config)
            if not snapshot.next:
                print(f"No run pending for thread {args.thread_id!r} -- pass --state-file.")
                return
            pending = snapshot.tasks[0].interrupts[0]
            print(json.dumps(pending.value, indent=2, default=str))
            response = _prompt_for_pending_response(pending)
            result = _run_streaming(graph, Command(resume=response), config)

        if "__interrupt__" in result:
            _print_interrupt(result["__interrupt__"][0], args.thread_id)
        else:
            _print_final(result)


if __name__ == "__main__":
    main()
