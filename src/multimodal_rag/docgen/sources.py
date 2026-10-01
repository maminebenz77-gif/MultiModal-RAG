"""Source resolution: turning a "reuse an existing tag" or "ingest a new
folder" choice into a SourceSpec, for each role a docgen run needs
(task_docs always, reference_kb only if the questions require a
comparison).

Deliberately has no input()/terminal code anywhere in this module --
resolve_source() takes an already-made SourceChoice value and returns a
SourceSpec, so it's testable with plain fixed inputs. The actual
terminal wizard (asking the question, listing tags via
docgen.tags.list_known_tags() as suggestions, reading what the user
typed) is CLI glue built separately in docgen/cli.py, alongside the
other human-in-the-loop prompts that module needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..identity import Principal
from ..metadata import Classification
from .ingest import ingest_folder
from .stack import DocgenStack
from .tags import Role, build_docgen_tag

SourceRole = Literal["task_docs", "reference_kb"]

_ROLE_FOR_SOURCE: dict[SourceRole, Role] = {"task_docs": "task", "reference_kb": "ref"}


@dataclass(frozen=True)
class SourceSpec:
    role: SourceRole
    tag: str
    required: bool


@dataclass(frozen=True)
class ReuseExisting:
    """Point a source at documents already in the corpus, by a tag the
    user picked or typed themselves -- never a generated id."""

    tag: str


@dataclass(frozen=True)
class IngestNew:
    """Ingest a new folder for this source; `label` becomes part of the
    tag docgen creates (see tags.build_docgen_tag), prefixed by role."""

    folder: Path
    label: str
    classification: Classification


SourceChoice = ReuseExisting | IngestNew


def resolve_source(
    role: SourceRole,
    choice: SourceChoice,
    stack: DocgenStack,
    *,
    required: bool = True,
    principal: Principal | None = None,
) -> SourceSpec:
    """Apply one role's reuse-or-ingest decision and return the
    resulting SourceSpec. An IngestNew choice actually ingests the
    folder now (this is deterministic, non-LLM work -- see the docgen
    build plan for why it runs before the graph, not as a graph node);
    a ReuseExisting choice does no I/O beyond trusting the tag as given."""
    if isinstance(choice, ReuseExisting):
        tag = choice.tag
    else:
        tag = build_docgen_tag(_ROLE_FOR_SOURCE[role], choice.label)
        ingest_folder(choice.folder, tag, choice.classification, stack, principal=principal)
    return SourceSpec(role=role, tag=tag, required=required)
