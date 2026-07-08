"""Disposable SQLite index (brief §2.6, §7).

Markdown is truth; this DB is a cache that ``reindex()`` rebuilds from scratch.
Besides mirroring the entities it materialises two denormalised tables that
power the core value (brief §5):

* ``intervention_effects`` — one row per (intervention, diagnostic, experiment)
  with the signed delta. Drives ``causal_map``.
* ``evaluation`` — one row per (experiment, predicted diagnostic) with
  predicted vs observed direction and a ``matched`` flag. Drives ``calibration``
  and the negative-result ranking in ``recall``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import compute_observed_effects, compute_prediction_match
from .storage import Vault

SCHEMA = """
CREATE TABLE problems (id TEXT PRIMARY KEY, title TEXT, description TEXT,
    status TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE hypotheses (id TEXT PRIMARY KEY, problem_id TEXT, statement TEXT,
    rationale TEXT, plan TEXT, status TEXT, supersedes TEXT, created_at TEXT);
CREATE TABLE experiments (id TEXT PRIMARY KEY, hypothesis_id TEXT, git_ref TEXT,
    external_run TEXT, verdict TEXT, prediction_overall INTEGER, closed INTEGER,
    notes TEXT, created_at TEXT);
CREATE TABLE diagnostics (id TEXT PRIMARY KEY, name TEXT, unit TEXT,
    direction TEXT, description TEXT);
CREATE TABLE interventions (id TEXT PRIMARY KEY, name TEXT, description TEXT);
CREATE TABLE papers (id TEXT PRIMARY KEY, title TEXT, arxiv_id_or_url TEXT);

CREATE TABLE tags (entity_id TEXT, entity_kind TEXT, tag TEXT, axis TEXT);
CREATE TABLE relations (src TEXT, src_kind TEXT, rel TEXT, dst TEXT, dst_kind TEXT);
CREATE TABLE measurements (experiment_id TEXT, diagnostic_id TEXT,
    before REAL, after REAL, delta REAL, direction TEXT);

CREATE TABLE intervention_effects (intervention_id TEXT, diagnostic_id TEXT,
    experiment_id TEXT, hypothesis_id TEXT, problem_id TEXT,
    delta REAL, direction TEXT, topic_tags TEXT, problem_tags TEXT);
CREATE TABLE evaluation (experiment_id TEXT, hypothesis_id TEXT, problem_id TEXT,
    diagnostic_id TEXT, predicted TEXT, observed TEXT, delta REAL,
    matched INTEGER, is_primary INTEGER, interventions TEXT,
    topic_tags TEXT, problem_tags TEXT);

CREATE INDEX ix_hyp_problem ON hypotheses(problem_id);
CREATE INDEX ix_exp_hyp ON experiments(hypothesis_id);
CREATE INDEX ix_tags ON tags(tag);
CREATE INDEX ix_meas_diag ON measurements(diagnostic_id);
CREATE INDEX ix_ie_diag ON intervention_effects(diagnostic_id);
CREATE INDEX ix_ie_int ON intervention_effects(intervention_id);
CREATE INDEX ix_eval_diag ON evaluation(diagnostic_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _csv(xs) -> str:
    return ",".join(xs)


def reindex(vault: Vault, db_path: str | Path) -> dict:
    """Rebuild the SQLite index from the markdown vault. Idempotent."""
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    conn = connect(db_path)
    conn.executescript(SCHEMA)

    problems = list(vault.all_problems())
    hyps = list(vault.all_hypotheses())
    exps = list(vault.all_experiments())
    diags = {d.id: d for d in vault.all_diagnostics()}
    ivs = list(vault.all_interventions())
    papers = list(vault.all_papers())
    hyp_by_id = {h.id: h for h in hyps}

    def add_tags(eid, kind, topic, problem):
        for t in topic:
            conn.execute("INSERT INTO tags VALUES (?,?,?,?)", (eid, kind, t, "topic"))
        for t in problem:
            conn.execute("INSERT INTO tags VALUES (?,?,?,?)", (eid, kind, t, "problem"))

    def add_rel(src, src_kind, rel, dst, dst_kind):
        conn.execute("INSERT INTO relations VALUES (?,?,?,?,?)",
                     (src, src_kind, rel, dst, dst_kind))

    # --- entities ---
    for p in problems:
        conn.execute("INSERT INTO problems VALUES (?,?,?,?,?,?)",
                     (p.id, p.title, p.description, p.status.value, p.created_at, p.updated_at))
        add_tags(p.id, "problem", p.topic_tags, p.problem_tags)
        for r in p.paper_refs:
            add_rel(p.id, "problem", "cites", r, "paper")

    for h in hyps:
        conn.execute("INSERT INTO hypotheses VALUES (?,?,?,?,?,?,?,?)",
                     (h.id, h.problem_id, h.statement, h.rationale, h.plan,
                      h.status.value, h.supersedes, h.created_at))
        add_tags(h.id, "hypothesis", h.topic_tags, h.problem_tags)
        add_rel(h.problem_id, "problem", "has", h.id, "hypothesis")
        for iv in h.interventions:
            add_rel(h.id, "hypothesis", "applies", iv, "intervention")
        for pe in h.predicted_effects:
            add_rel(h.id, "hypothesis", "predicts", pe.diagnostic_id, "diagnostic")
        for r in h.paper_refs:
            add_rel(h.id, "hypothesis", "cites", r, "paper")
        if h.supersedes:
            add_rel(h.id, "hypothesis", "supersedes", h.supersedes, "hypothesis")

    for d in diags.values():
        conn.execute("INSERT INTO diagnostics VALUES (?,?,?,?,?)",
                     (d.id, d.name, d.unit, d.direction.value, d.description))
    for iv in ivs:
        conn.execute("INSERT INTO interventions VALUES (?,?,?)",
                     (iv.id, iv.name, iv.description))
        add_tags(iv.id, "intervention", iv.topic_tags, [])
    for pa in papers:
        conn.execute("INSERT INTO papers VALUES (?,?,?)",
                     (pa.id, pa.title, pa.arxiv_id_or_url))
        add_tags(pa.id, "paper", pa.topic_tags, [])

    # --- experiments + derived tables ---
    for e in exps:
        conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?)",
                     (e.id, e.hypothesis_id, e.git_ref, e.external_run,
                      e.verdict.value if e.verdict else None,
                      int(e.prediction_match.overall) if e.prediction_match else None,
                      int(e.closed), e.notes, e.created_at))
        add_rel(e.hypothesis_id, "hypothesis", "tested_by", e.id, "experiment")
        if e.git_ref:
            add_rel(e.id, "experiment", "at_commit", e.git_ref, "git")

        h = hyp_by_id.get(e.hypothesis_id)
        topic = h.topic_tags if h else []
        problem_tags = h.problem_tags if h else []
        problem_id = h.problem_id if h else ""
        interventions = h.interventions if h else []
        for iv in interventions:
            add_rel(e.id, "experiment", "applies", iv, "intervention")

        # observed effects (recompute from measurements = truth)
        observed = compute_observed_effects(e.diagnostics_measured)
        obs_by = {o.diagnostic_id: o for o in observed}
        for m in e.diagnostics_measured:
            o = obs_by[m.diagnostic_id]
            conn.execute("INSERT INTO measurements VALUES (?,?,?,?,?,?)",
                         (e.id, m.diagnostic_id, m.before, m.after, o.delta, o.direction.value))
            add_rel(e.id, "experiment", "measures", m.diagnostic_id, "diagnostic")
            # intervention_effects: attribute the delta to each applied intervention
            for iv in interventions:
                conn.execute(
                    "INSERT INTO intervention_effects VALUES (?,?,?,?,?,?,?,?,?)",
                    (iv, m.diagnostic_id, e.id, e.hypothesis_id, problem_id,
                     o.delta, o.direction.value, _csv(topic), _csv(problem_tags)),
                )

        # evaluation rows (predicted vs observed) — only meaningful once closed
        if h and e.closed:
            pm = compute_prediction_match(h.predicted_effects, observed)
            primary_id = h.predicted_effects[0].diagnostic_id if h.predicted_effects else None
            for pe in h.predicted_effects:
                o = obs_by.get(pe.diagnostic_id)
                conn.execute(
                    "INSERT INTO evaluation VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (e.id, h.id, problem_id, pe.diagnostic_id, pe.direction.value,
                     o.direction.value if o else None,
                     o.delta if o else None,
                     int(pm.per_diagnostic.get(pe.diagnostic_id, False)),
                     int(pe.diagnostic_id == primary_id),
                     _csv(interventions), _csv(topic), _csv(problem_tags)),
                )

    conn.commit()
    counts = {
        k: conn.execute(f"SELECT COUNT(*) FROM {k}").fetchone()[0]
        for k in ("problems", "hypotheses", "experiments", "diagnostics",
                  "interventions", "papers", "measurements",
                  "intervention_effects", "evaluation")
    }
    conn.close()
    return counts
