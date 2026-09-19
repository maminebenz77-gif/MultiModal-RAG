"""Runs every data/eval/<expertise>/ folder as a native Langfuse
Experiment -- one dataset, and one dataset run, per expertise -- so "is
legal's correctness holding steady week over week" is a comparison
Langfuse's own UI renders natively (per-run averages, run-vs-run
diffing), not just a number run_expert_eval.py prints once and forgets.

Same relationship to run_expert_eval.py that langfuse_experiment.py has
to run_eval.py: reuses this project's own scoring (judge.py) and the
exact same real AgentChain construction (build_expertise_agent) as the
actual evaluation logic -- Langfuse's run_experiment() supplies the
harness (dataset sync, per-item tracing, score attachment, run
comparison), not the scoring itself. A no-op, loudly, if Langfuse isn't
configured -- this script's whole point is sending data to Langfuse.

One dataset per expertise, not one shared dataset with an expertise
field, because each expertise already gets its own isolated ingestion
collection (see build_expertise_agent) -- keeping the datasets separate
too means a question in one expertise's set can never show up mixed into
another's run history.

Run: `uv run python -m multimodal_rag.evaluation.langfuse_expert_eval`
"""

from typing import Any

from langfuse import Evaluation, Langfuse
from langfuse.experiment import EvaluatorFunction

from ..generation.agent import AgentChain
from ..tracing import get_langfuse_client
from .judge import JudgeParseError, score_answer_correctness
from .run_expert_eval import _EVAL_ROOT, _discover_expertise_dirs, _load_qa, build_expertise_agent

_DATASET_PREFIX = "expert-eval-"
_EXPERIMENT_NAME_PREFIX = "Expert eval: "


def _dataset_name(expertise: str) -> str:
    return f"{_DATASET_PREFIX}{expertise}"


def _sync_dataset(client: Langfuse, expertise: str, qa_items: list[dict[str, Any]]) -> None:
    """Both create_dataset and create_dataset_item upsert by name/id (see
    langfuse_experiment.py's _sync_dataset) -- safe to call on every run,
    and an edit to qa.json (a new question, a corrected expert_answer)
    shows up here on the next run too, not just a one-time upload.
    """
    dataset_name = _dataset_name(expertise)
    client.create_dataset(
        name=dataset_name,
        description=(
            f"Expert-authored Q&A for the '{expertise}' expertise (data/eval/{expertise}/qa.json)."
        ),
    )
    for item in qa_items:
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item["id"],
            input=item["question"],
            expected_output={
                "expert_answer": item["expert_answer"],
                "expect_refusal": item.get("expect_refusal", False),
            },
        )


def _make_task(agent: AgentChain) -> Any:
    def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
        rag_answer = agent.answer(item.input)
        return {"answer": rag_answer.answer, "refused": rag_answer.refused}

    return task


def _correctness_evaluator(
    *, input: Any, output: dict[str, Any], expected_output: dict[str, Any], **kwargs: Any
) -> list[Evaluation]:
    if expected_output.get("expect_refusal"):
        # Scored by _refusal_accuracy_evaluator instead -- see its
        # docstring for why a free-text judge is the wrong tool here.
        return []
    if output["refused"]:
        # A false refusal: an answerable item the agent declined anyway.
        # A real failure, but not one a free-text correctness judge can
        # honestly score -- there's no substantive answer to compare.
        return [Evaluation(name="false_refusal", value=True, data_type="BOOLEAN")]
    try:
        score = score_answer_correctness(input, output["answer"], expected_output["expert_answer"])
    except JudgeParseError:
        return []
    return [Evaluation(name="correctness", value=score)]


def _refusal_accuracy_evaluator(
    *, output: dict[str, Any], expected_output: dict[str, Any], **kwargs: Any
) -> list[Evaluation]:
    """Deliberately separate from correctness -- grading a refusal's
    ANSWER TEXT against expert_answer with a free-text judge is exactly
    the bug caught live in this project: a correct, terse "I don't know"
    scored 0 against a longer reference explanation. This only checks
    whether the agent refused at all, never what it said.
    """
    if not expected_output.get("expect_refusal"):
        return []
    correct = bool(output["refused"])
    return [Evaluation(name="refusal_accuracy", value=1.0 if correct else 0.0)]


_EVALUATORS: list[EvaluatorFunction] = [_correctness_evaluator, _refusal_accuracy_evaluator]


def main() -> None:
    client = get_langfuse_client()
    if client is None:
        print(
            "Langfuse isn't configured (see tracing.py) -- nothing to send. Set "
            "LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST in .env.local, or run "
            "`uv run python -m multimodal_rag.evaluation.run_expert_eval` for the "
            "console-only version instead."
        )
        return

    expertise_dirs = _discover_expertise_dirs()
    if not expertise_dirs:
        print(
            f"No expertise folders found under {_EVAL_ROOT}. See "
            f"{_EVAL_ROOT / 'README.md'} for the folder convention."
        )
        return

    for expertise_dir in expertise_dirs:
        name = expertise_dir.name
        qa_items = _load_qa(expertise_dir)
        _sync_dataset(client, name, qa_items)
        dataset = client.get_dataset(_dataset_name(name))

        agent = build_expertise_agent(expertise_dir)
        result = dataset.run_experiment(
            name=f"{_EXPERIMENT_NAME_PREFIX}{name}",
            run_name=name,
            task=_make_task(agent),
            evaluators=_EVALUATORS,
        )
        print(result.format())

    client.flush()


if __name__ == "__main__":
    main()
