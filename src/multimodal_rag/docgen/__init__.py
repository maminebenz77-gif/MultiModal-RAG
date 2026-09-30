"""Autonomous document-generation workflow, built on LangGraph.

Separate from generation/agent.py's AgentChain (the existing
conversational agent) on purpose -- see docs/technical-decisions.md and
the docgen build plan for why: this is a multi-minute autonomous job
with its own confirm/escalate/review checkpoints, launched on its own
(for now, docgen/cli.py), not a chat turn. It shares the underlying
ingestion pipeline, stores and providers with the rest of the app, but
never imports from or is imported by generation/agent.py.
"""
