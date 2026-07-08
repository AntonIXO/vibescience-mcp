"""recall, causal_map, calibration + index rebuild + Obsidian compat (brief §12)."""

import frontmatter

from vibescience_mcp.core import Store


# --------------------------------------------------------------------------- #
# recall: negative results first (brief §3, §8, §12)
# --------------------------------------------------------------------------- #
def test_recall_surfaces_refuted_above_confirmed(seeded):
    r = seeded.recall(problem_id="soft-token-attention-collapse")
    results = r["results"]
    assert len(results) >= 2
    # the refuted sibling must rank above the merely-confirmed one
    ids = [x["hypothesis_id"] for x in results]
    i_refuted = ids.index("rmsnorm-logscale-alone-fixes-collapse")
    i_confirmed = ids.index("projector-freeze-blocks-caption-grad")
    assert i_refuted < i_confirmed
    # the top result carries a failure reason with the killing delta
    top = results[0]
    assert top["is_negative"] is True
    assert top["why_it_failed"] is not None
    assert "attn_mass_soft_tokens" in top["why_it_failed"]


def test_recall_calibration_note_on_tracked_diagnostic(seeded):
    r = seeded.recall(query="attn_mass_soft_tokens collapse")
    assert r["calibration_note"] is not None
    assert "attn_mass_soft_tokens" in r["calibration_note"]


def test_recall_empty_on_unrelated_query(seeded):
    r = seeded.recall(query="completely unrelated topic xyzzy", topic_tags=["nonexistent"])
    assert r["results"] == []


# --------------------------------------------------------------------------- #
# causal_map (brief §5, §12)
# --------------------------------------------------------------------------- #
def test_causal_map_aggregates(seeded):
    cm = seeded.causal_map(problem_id="soft-token-attention-collapse")
    edges = {(e["intervention"], e["diagnostic"]): e for e in cm["edges"]}

    # unfreeze-projector moved attn_mass up by +0.29
    up = edges[("unfreeze-projector", "attn_mass_soft_tokens")]
    assert up["sign"] == "↑"
    assert abs(up["mean_delta"] - 0.29) < 1e-9
    assert up["up"] == 1 and up["down"] == 0

    # rmsnorm-logscale-init moved attn_mass DOWN (the refuted path)
    dn = edges[("rmsnorm-logscale-init", "attn_mass_soft_tokens")]
    assert dn["sign"] == "↓"
    assert dn["mean_delta"] < 0


def test_causal_map_canvas_emitted(seeded):
    path = seeded.write_canvas(problem_id="soft-token-attention-collapse")
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text())
    assert "nodes" in data and "edges" in data
    assert len(data["nodes"]) >= 2


# --------------------------------------------------------------------------- #
# calibration (brief §5, §12)
# --------------------------------------------------------------------------- #
def test_calibration_overall(seeded):
    # attn_mass_soft_tokens: predicted up twice; right once (confirming), wrong once (refuted)
    cal = seeded.calibration(diagnostic_id="attn_mass_soft_tokens")
    assert cal["n"] == 2
    assert cal["right"] == 1
    assert cal["accuracy"] == 0.5


def test_calibration_by_intervention(seeded):
    cal = seeded.calibration(intervention_id="unfreeze-projector")
    # both predicted effects of the confirming hyp matched
    assert cal["right"] == cal["n"]
    assert cal["accuracy"] == 1.0


# --------------------------------------------------------------------------- #
# index rebuild (brief §6, §12): reindex reproduces identical results
# --------------------------------------------------------------------------- #
def test_reindex_reproduces_results(seeded, vault):
    cm_before = seeded.causal_map(problem_id="soft-token-attention-collapse")
    cal_before = seeded.calibration(diagnostic_id="attn_mass_soft_tokens")
    recall_before = seeded.recall(problem_id="soft-token-attention-collapse")

    # nuke the SQLite index entirely
    seeded.db_path.unlink()
    assert not seeded.db_path.exists()

    # a fresh Store over the same markdown must rebuild identical answers
    s2 = Store(vault)
    assert s2.causal_map(problem_id="soft-token-attention-collapse")["edges"] == cm_before["edges"]
    assert s2.calibration(diagnostic_id="attn_mass_soft_tokens") == cal_before
    ids_before = [x["hypothesis_id"] for x in recall_before["results"]]
    ids_after = [x["hypothesis_id"] for x in s2.recall(problem_id="soft-token-attention-collapse")["results"]]
    assert ids_before == ids_after


# --------------------------------------------------------------------------- #
# Obsidian compatibility (brief §12): every entity is valid frontmatter markdown
# --------------------------------------------------------------------------- #
def test_every_entity_is_valid_frontmatter(seeded):
    root = seeded.vault.root
    md_files = list(root.rglob("*.md"))
    assert md_files, "no markdown written"
    for f in md_files:
        post = frontmatter.load(f)
        assert "id" in post.metadata, f"{f} missing id in frontmatter"
        assert post.content.strip(), f"{f} has empty body"


def test_wikilinks_present_for_graph(seeded):
    root = seeded.vault.root
    exp = (root / "experiments" / "unfreeze-projector-joint-loss.md").read_text()
    # measurements reference diagnostics as wikilinks; hypothesis is linked
    assert "[[attn_mass_soft_tokens]]" in exp
    assert "[[projector-freeze-blocks-caption-grad]]" in exp
    hyp = (root / "hypotheses" / "projector-freeze-blocks-caption-grad.md").read_text()
    assert "[[unfreeze-projector]]" in hyp
    assert "[[soft-token-attention-collapse]]" in hyp
