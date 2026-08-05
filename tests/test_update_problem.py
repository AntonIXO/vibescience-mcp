"""update_problem: amending a record after evidence changes its framing.

The motivating failure: a problem's stated symptom gets root-caused and fixed,
but the record still asserts the pre-fix symptom. `recall` then hands a future
session the confirmed fix AND a description contradicting it. Before this tool
existed the only remedy was hand-editing the vault markdown and reindexing,
which bypassed tag validation and silently left `updated_at` stale.
"""

from __future__ import annotations

import pytest

from vibescience_mcp import core
from vibescience_mcp.core import Store, VibeScienceError
from vibescience_mcp.models import ProblemStatus

PID = "soft-token-attention-collapse"


# --------------------------------------------------------------------- #
# updated_at — the write-once bug
# --------------------------------------------------------------------- #
def test_create_stamps_updated_at_equal_to_created_at(seeded):
    p = seeded.get_problem(PID)
    assert p.updated_at == p.created_at


def test_update_bumps_updated_at_and_leaves_created_at_alone(seeded, monkeypatch):
    """now_iso() is deliberately second-resolution for readable vault files, so
    a same-second edit is indistinguishable by wall clock. Pin the clock forward
    to prove the bump is actually performed rather than coincidental."""
    before = seeded.get_problem(PID)
    monkeypatch.setattr(core, "now_iso", lambda: "2099-01-01T00:00:00+00:00")
    after = seeded.update_problem(PID, description="re-scoped after root cause")
    assert after.created_at == before.created_at
    assert after.updated_at == "2099-01-01T00:00:00+00:00"
    assert after.updated_at > before.updated_at


def test_updated_at_survives_the_markdown_roundtrip(seeded):
    """The vault is the source of truth, so the bump must be persisted, not
    just returned from the in-memory object."""
    bumped = seeded.update_problem(PID, description="persisted?")
    assert seeded.get_problem(PID).updated_at == bumped.updated_at


def test_link_paper_also_bumps_updated_at(seeded, monkeypatch):
    """link_paper mutates the problem, so it must not leave a stale stamp."""
    before = seeded.get_problem(PID).updated_at
    seeded.add_paper("Prefix-Tuning", arxiv_id_or_url="2101.00190", id="prefix-tuning")
    monkeypatch.setattr(core, "now_iso", lambda: "2099-01-01T00:00:00+00:00")
    seeded.link_paper(PID, "prefix-tuning")
    assert seeded.get_problem(PID).updated_at == "2099-01-01T00:00:00+00:00"
    assert seeded.get_problem(PID).updated_at > before


def test_read_does_not_fabricate_an_edit_when_stamp_is_missing(seeded, vault):
    """A hand-edited or pre-schema file with no updated_at must fall back to
    created_at. Falling through to the model default would stamp *now* on a
    pure read and invent an edit that never happened."""
    import frontmatter

    path = vault / "problems" / f"{PID}.md"
    post = frontmatter.load(path)
    del post.metadata["updated_at"]
    path.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")

    p = Store(vault).get_problem(PID)
    assert p.updated_at == p.created_at


# --------------------------------------------------------------------- #
# field amendment
# --------------------------------------------------------------------- #
def test_only_passed_fields_change(seeded):
    before = seeded.get_problem(PID)
    after = seeded.update_problem(PID, description="new text")
    assert after.description == "new text"
    assert after.title == before.title
    assert after.status == before.status
    assert after.topic_tags == before.topic_tags
    assert after.problem_tags == before.problem_tags
    assert after.paper_refs == before.paper_refs


def test_description_amendment_is_searchable_after_reindex(seeded):
    """update_problem reindexes, so recall must see the new wording — that is
    the entire point of amending rather than leaving a stale record."""
    seeded.update_problem(
        PID, description="SUPERSEDED: root cause was a train/eval mask inversion."
    )
    hits = seeded.recall(query="train/eval mask inversion")
    ids = [c["id"] for c in hits["context"] if c["kind"] == "problem"]
    assert PID in ids


def test_tags_go_through_the_registry(seeded):
    """Hand-editing markdown bypassed tag validation; the tool must not."""
    with pytest.raises(VibeScienceError, match="Unknown problem tag"):
        seeded.update_problem(PID, problem_tags=["not-a-registered-tag"])


def test_tag_aliases_resolve_on_update(seeded):
    p = seeded.update_problem(PID, problem_tags=["frozen-projector"])
    assert p.problem_tags == ["projector-freeze"]


def test_unknown_paper_ref_is_rejected(seeded):
    with pytest.raises(VibeScienceError, match="Unknown paper"):
        seeded.update_problem(PID, paper_refs=["no-such-paper"])


def test_unknown_problem_is_rejected(seeded):
    with pytest.raises(VibeScienceError, match="Unknown problem"):
        seeded.update_problem("no-such-problem", description="x")


# --------------------------------------------------------------------- #
# status — resolution must be earned, not asserted
# --------------------------------------------------------------------- #
def test_resolve_allowed_when_a_confirmed_hypothesis_exists(seeded):
    """The seeded problem has a confirmed hypothesis, so it has earned this."""
    p = seeded.update_problem(PID, status="resolved")
    assert p.status is ProblemStatus.resolved
    assert seeded.list_problems(status="resolved")[0].id == PID


def test_resolve_blocked_without_a_confirmed_hypothesis(seeded):
    """Same discipline as close_experiment: a verdict is computed from
    evidence, never declared."""
    seeded.create_problem("unearned", id="unearned",
                          description="no hypothesis has been confirmed here")
    with pytest.raises(VibeScienceError, match="no CONFIRMED hypothesis"):
        seeded.update_problem("unearned", status="resolved")
    assert seeded.get_problem("unearned").status is ProblemStatus.open


def test_parked_is_ungated(seeded):
    """Parking states something about your attention, not about the evidence."""
    seeded.create_problem("shelved", id="shelved", description="not pursuing this now")
    assert seeded.update_problem("shelved", status="parked").status is ProblemStatus.parked


def test_resolved_problem_can_be_reopened_and_reresolved(seeded):
    seeded.update_problem(PID, status="resolved")
    assert seeded.update_problem(PID, status="open").status is ProblemStatus.open
    assert seeded.update_problem(PID, status="resolved").status is ProblemStatus.resolved


def test_invalid_status_is_rejected(seeded):
    with pytest.raises(ValueError):
        seeded.update_problem(PID, status="mostly-fixed")


# --------------------------------------------------------------------- #
# MCP transport
# --------------------------------------------------------------------- #
def test_exposed_as_an_mcp_tool():
    from vibescience_mcp import server

    assert hasattr(server, "update_problem")


def _unwrap(res):
    """FastMCP returns a ToolResult. A dict-returning tool is surfaced directly
    as structured_content; other shapes get wrapped under a 'result' key."""
    sc = res.structured_content
    return sc["result"] if "result" in sc else sc


def test_mcp_roundtrip_amends_and_persists(seeded, vault, monkeypatch):
    """End-to-end through the MCP transport, not just the Store: the tool must
    amend, bump the stamp, and persist to the vault."""
    import asyncio

    from vibescience_mcp import server

    # store() is a module-global singleton built from VAULT at import time;
    # point it at this test's vault instead.
    monkeypatch.setattr(server, "_store", seeded)

    async def go():
        r = await server.mcp.call_tool(
            "update_problem",
            {"id": PID, "status": "resolved", "description": "RESOLVED: root caused."},
        )
        got = await server.mcp.call_tool("get_problem", {"id": PID})
        return _unwrap(r), _unwrap(got)

    updated, fetched = asyncio.run(go())

    assert updated["ok"] is True
    assert updated["data"]["status"] == "resolved"
    assert updated["data"]["description"] == "RESOLVED: root caused."
    assert fetched["data"]["status"] == "resolved"


def test_mcp_errors_are_returned_not_raised(seeded, vault, monkeypatch):
    """An unregistered tag must come back as a structured error payload."""
    import asyncio

    from vibescience_mcp import server

    # store() is a module-global singleton built from VAULT at import time;
    # point it at this test's vault instead.
    monkeypatch.setattr(server, "_store", seeded)

    async def go():
        return _unwrap(await server.mcp.call_tool(
            "update_problem", {"id": PID, "problem_tags": ["bogus-tag"]}))

    out = asyncio.run(go())
    assert out["ok"] is False
    assert "Unknown problem tag" in out["error"]
