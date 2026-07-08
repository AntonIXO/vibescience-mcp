"""Domain logic: lifecycle, recall, causal_map, calibration (brief §5, §6, §8, §9).

This layer is transport-agnostic — ``server.py`` wraps it as MCP tools. It owns
the scientific workflow rules: a hypothesis needs a committed prediction, the
verdict is *computed* not asserted, and negative results are up-weighted in
recall.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .index import connect, reindex
from .models import (
    Diagnostic,
    DiagDirection,
    Direction,
    Experiment,
    Hypothesis,
    HypothesisStatus,
    Intervention,
    Measurement,
    Paper,
    PredictedEffect,
    Problem,
    ProblemStatus,
    Verdict,
    compute_observed_effects,
    compute_prediction_match,
    compute_verdict,
    now_iso,
    slugify,
    verdict_to_hypothesis_status,
)
from .storage import Vault


class VibeScienceError(Exception):
    """Domain-level validation error, surfaced to the agent as a clean message."""


class Store:
    def __init__(self, vault_root: str | Path):
        self.vault = Vault(vault_root)
        self.db_path = self.vault.root / ".index.sqlite"
        self.reindex()

    # ------------------------------------------------------------------ #
    # Index
    # ------------------------------------------------------------------ #
    def reindex(self) -> dict:
        return reindex(self.vault, self.db_path)

    def _conn(self):
        if not self.db_path.exists():
            self.reindex()
        return connect(self.db_path)

    # ------------------------------------------------------------------ #
    # Registry: diagnostics + interventions + papers
    # ------------------------------------------------------------------ #
    def register_diagnostic(self, name, unit="", direction="neutral",
                            description="", id=None) -> Diagnostic:
        d = Diagnostic(id=id or slugify(name), name=name, unit=unit,
                       direction=DiagDirection(direction), description=description)
        self.vault.write_diagnostic(d)
        self.reindex()
        return d

    def list_diagnostics(self) -> list[Diagnostic]:
        return list(self.vault.all_diagnostics())

    def _diag_ids(self) -> set[str]:
        return set(self.vault.ids("diagnostics"))

    def register_intervention(self, name, description="", topic_tags=None, id=None) -> Intervention:
        iv = Intervention(id=id or slugify(name), name=name, description=description,
                          topic_tags=topic_tags or [])
        self.vault.write_intervention(iv)
        self.reindex()
        return iv

    def list_interventions(self) -> list[Intervention]:
        return list(self.vault.all_interventions())

    def add_paper(self, title, arxiv_id_or_url="", key_claims=None, topic_tags=None, id=None) -> Paper:
        p = Paper(id=id or slugify(title), title=title, arxiv_id_or_url=arxiv_id_or_url,
                  key_claims=key_claims or [], topic_tags=topic_tags or [])
        self.vault.write_paper(p)
        self.reindex()
        return p

    def link_paper(self, entity_id: str, paper_id: str) -> str:
        if not self.vault.exists("papers", paper_id):
            raise VibeScienceError(f"Unknown paper '{paper_id}'.")
        if self.vault.exists("problems", entity_id):
            e = self.vault.read_problem(entity_id)
            if paper_id not in e.paper_refs:
                e.paper_refs.append(paper_id)
            e.updated_at = now_iso()
            self.vault.write_problem(e)
        elif self.vault.exists("hypotheses", entity_id):
            e = self.vault.read_hypothesis(entity_id)
            if paper_id not in e.paper_refs:
                e.paper_refs.append(paper_id)
            self.vault.write_hypothesis(e)
        else:
            raise VibeScienceError(f"Unknown entity '{entity_id}' to link a paper to.")
        self.reindex()
        return f"Linked [[{paper_id}]] to [[{entity_id}]]."

    # ------------------------------------------------------------------ #
    # Problem
    # ------------------------------------------------------------------ #
    def create_problem(self, title, description="", topic_tags=None, problem_tags=None,
                       paper_refs=None, id=None) -> Problem:
        p = Problem(id=id or slugify(title), title=title, description=description,
                    topic_tags=topic_tags or [], problem_tags=problem_tags or [],
                    paper_refs=paper_refs or [])
        if self.vault.exists("problems", p.id):
            raise VibeScienceError(f"Problem '{p.id}' already exists.")
        self.vault.write_problem(p)
        self.reindex()
        return p

    def get_problem(self, id: str) -> Problem:
        if not self.vault.exists("problems", id):
            raise VibeScienceError(f"Unknown problem '{id}'.")
        return self.vault.read_problem(id)

    def list_problems(self, status=None, tags=None) -> list[Problem]:
        out = list(self.vault.all_problems())
        if status:
            out = [p for p in out if p.status.value == status]
        if tags:
            tagset = set(tags)
            out = [p for p in out if tagset & set(p.topic_tags + p.problem_tags)]
        return out

    # ------------------------------------------------------------------ #
    # Hypothesis  (prediction is mandatory — brief §6)
    # ------------------------------------------------------------------ #
    def propose_hypothesis(self, problem_id, statement, rationale="", interventions=None,
                           predicted_effects=None, plan="", topic_tags=None,
                           problem_tags=None, papers=None, supersedes=None, id=None) -> Hypothesis:
        if not self.vault.exists("problems", problem_id):
            raise VibeScienceError(f"Unknown problem '{problem_id}'. Create it first.")
        predicted_effects = predicted_effects or []
        if not predicted_effects:
            raise VibeScienceError(
                "A hypothesis MUST commit >=1 predicted_effect on a registered "
                "diagnostic BEFORE testing (brief §6). Predict, then test."
            )
        known = self._diag_ids()
        pes: list[PredictedEffect] = []
        for pe in predicted_effects:
            pe = PredictedEffect.model_validate(pe)
            if pe.diagnostic_id not in known:
                raise VibeScienceError(
                    f"predicted_effect references unregistered diagnostic "
                    f"'{pe.diagnostic_id}'. register_diagnostic() first "
                    f"(diagnostics are a fixed basis, brief §4)."
                )
            pes.append(pe)
        for iv in interventions or []:
            if not self.vault.exists("interventions", iv):
                raise VibeScienceError(
                    f"Unknown intervention '{iv}'. register_intervention() first."
                )
        h = Hypothesis(
            id=id or slugify(statement)[:80], problem_id=problem_id, statement=statement,
            rationale=rationale, interventions=interventions or [], predicted_effects=pes,
            plan=plan, topic_tags=topic_tags or [], problem_tags=problem_tags or [],
            paper_refs=papers or [], supersedes=supersedes,
        )
        if self.vault.exists("hypotheses", h.id):
            raise VibeScienceError(f"Hypothesis '{h.id}' already exists; pass a distinct id.")
        self.vault.write_hypothesis(h)
        self.reindex()
        return h

    def get_hypothesis(self, id: str) -> Hypothesis:
        if not self.vault.exists("hypotheses", id):
            raise VibeScienceError(f"Unknown hypothesis '{id}'.")
        return self.vault.read_hypothesis(id)

    # ------------------------------------------------------------------ #
    # Experiment lifecycle
    # ------------------------------------------------------------------ #
    def _git_ref(self) -> str:
        """Best-effort current branch@shortsha via subprocess (brief §9)."""
        try:
            root = self.vault.root
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=root, timeout=5,
            ).stdout.strip()
            sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=root, timeout=5,
            ).stdout.strip()
            if branch and sha:
                return f"{branch}@{sha}"
        except Exception:
            pass
        return ""

    def start_experiment(self, hypothesis_id, git_ref=None, external_run="",
                         config_note="", id=None) -> Experiment:
        if not self.vault.exists("hypotheses", hypothesis_id):
            raise VibeScienceError(f"Unknown hypothesis '{hypothesis_id}'.")
        if not git_ref:
            git_ref = self._git_ref()
        eid = id or slugify(f"{hypothesis_id}-exp-{now_iso()[:19]}")
        e = Experiment(id=eid, hypothesis_id=hypothesis_id, git_ref=git_ref or "",
                       external_run=external_run, config_note=config_note)
        self.vault.write_experiment(e)
        # move the hypothesis into 'testing'
        h = self.vault.read_hypothesis(hypothesis_id)
        if h.status in (HypothesisStatus.proposed,):
            h.status = HypothesisStatus.testing
            self.vault.write_hypothesis(h)
        self.reindex()
        return e

    def record_diagnostics(self, experiment_id, measurements) -> Experiment:
        if not self.vault.exists("experiments", experiment_id):
            raise VibeScienceError(f"Unknown experiment '{experiment_id}'.")
        e = self.vault.read_experiment(experiment_id)
        if e.closed:
            raise VibeScienceError(f"Experiment '{experiment_id}' is closed; reopen not supported.")
        known = self._diag_ids()
        existing = {m.diagnostic_id: m for m in e.diagnostics_measured}
        for m in measurements:
            m = Measurement.model_validate(m)
            if m.diagnostic_id not in known:
                raise VibeScienceError(
                    f"Measurement references unregistered diagnostic '{m.diagnostic_id}'."
                )
            existing[m.diagnostic_id] = m
        e.diagnostics_measured = list(existing.values())
        e.observed_effects = compute_observed_effects(e.diagnostics_measured)
        h = self.vault.read_hypothesis(e.hypothesis_id)
        pred = {pe.diagnostic_id: pe.direction for pe in h.predicted_effects}
        self.vault.write_experiment(e, predicted=pred)
        self.reindex()
        return e

    def close_experiment(self, experiment_id, notes="") -> dict:
        """Compute observed_effects, prediction_match, verdict; propagate status.

        Never auto-commits git — returns a *suggested* next action (brief §9).
        """
        if not self.vault.exists("experiments", experiment_id):
            raise VibeScienceError(f"Unknown experiment '{experiment_id}'.")
        e = self.vault.read_experiment(experiment_id)
        if not e.diagnostics_measured:
            raise VibeScienceError(
                "Cannot close: no diagnostics recorded. record_diagnostics() first."
            )
        h = self.vault.read_hypothesis(e.hypothesis_id)
        e.observed_effects = compute_observed_effects(e.diagnostics_measured)
        e.prediction_match = compute_prediction_match(h.predicted_effects, e.observed_effects)
        e.verdict = compute_verdict(h.predicted_effects, e.observed_effects)
        e.closed = True
        if notes:
            e.notes = notes
        pred = {pe.diagnostic_id: pe.direction for pe in h.predicted_effects}
        self.vault.write_experiment(e, predicted=pred)
        # propagate to hypothesis
        h.status = verdict_to_hypothesis_status(e.verdict)
        self.vault.write_hypothesis(h)
        self.reindex()

        # suggested next action (never performed)
        suggestion = None
        if e.prediction_match.overall and e.verdict == Verdict.supports:
            ref = e.git_ref or "<current HEAD>"
            suggestion = f"prediction matched → consider committing `{ref}`"
        elif e.verdict == Verdict.refutes:
            suggestion = ("prediction refuted → record why it failed; recall() will "
                          "surface this so it is never silently retried")
        else:
            suggestion = "inconclusive → the primary diagnostic didn't move; revise the hypothesis"

        return {
            "experiment_id": e.id,
            "verdict": e.verdict.value,
            "prediction_match": e.prediction_match.model_dump(),
            "observed_effects": [o.model_dump() for o in e.observed_effects],
            "hypothesis_status": h.status.value,
            "suggested_next_action": suggestion,
        }

    # ------------------------------------------------------------------ #
    # RECALL — negative-first pre-mortem gate (brief §6, §8)
    # ------------------------------------------------------------------ #
    def recall(self, query=None, topic_tags=None, problem_tags=None,
               problem_id=None, limit=10) -> dict:
        conn = self._conn()
        q = (query or "").lower()
        topic_tags = set(topic_tags or [])
        problem_tags = set(problem_tags or [])

        rows = conn.execute(
            "SELECT h.id, h.problem_id, h.statement, h.rationale, h.status "
            "FROM hypotheses h"
        ).fetchall()

        def tags_for(hid):
            tt = conn.execute(
                "SELECT tag, axis FROM tags WHERE entity_id=? AND entity_kind='hypothesis'",
                (hid,),
            ).fetchall()
            topic = {r["tag"] for r in tt if r["axis"] == "topic"}
            prob = {r["tag"] for r in tt if r["axis"] == "problem"}
            return topic, prob

        scored = []
        for r in rows:
            hid = r["id"]
            topic, prob = tags_for(hid)
            score = 0.0
            reasons = []
            if problem_id and r["problem_id"] == problem_id:
                score += 3.0
                reasons.append("same problem")
            tt_hit = topic_tags & topic
            pt_hit = problem_tags & prob
            if tt_hit:
                score += 2.0 * len(tt_hit); reasons.append("topic:" + ",".join(sorted(tt_hit)))
            if pt_hit:
                score += 2.0 * len(pt_hit); reasons.append("problem:" + ",".join(sorted(pt_hit)))
            if q:
                hay = (r["statement"] + " " + (r["rationale"] or "")).lower()
                if q in hay:
                    score += 1.5; reasons.append("keyword")
                else:
                    words = [w for w in q.split() if len(w) > 2]
                    hits = sum(1 for w in words if w in hay)
                    if hits:
                        score += 0.5 * hits; reasons.append(f"keyword×{hits}")
            # Up-weight negative / null results (brief §3, §8) — but ONLY when
            # they are already relevant. The boost rides an existing relevance
            # signal ("sort refuted matches to the top *when they're relevant*",
            # §8); it must not manufacture relevance for an unrelated query.
            status = r["status"]
            if score <= 0:
                continue
            if status in ("refuted", "inconclusive"):
                score += 2.5

            # one-line failure reason + the killing delta
            why = None
            if status in ("refuted", "inconclusive"):
                ev = conn.execute(
                    "SELECT diagnostic_id, predicted, observed, delta, matched "
                    "FROM evaluation WHERE hypothesis_id=? AND is_primary=1 LIMIT 1",
                    (hid,),
                ).fetchone()
                if ev:
                    why = (f"{status}: predicted {ev['diagnostic_id']} {ev['predicted']}, "
                           f"observed {ev['observed']} (Δ {ev['delta']:+g})")
                else:
                    why = f"{status}: no confirming measurement"

            scored.append({
                "hypothesis_id": hid,
                "problem_id": r["problem_id"],
                "statement": r["statement"],
                "status": status,
                "score": round(score, 2),
                "match_reasons": reasons,
                "why_it_failed": why,
                "is_negative": status in ("refuted", "inconclusive"),
            })

        # negative results first, then by score
        scored.sort(key=lambda x: (x["is_negative"], x["score"]), reverse=True)
        results = scored[:limit]

        # calibration note if the query touches a tracked diagnostic/tag
        cal_note = self._recall_calibration_note(conn, topic_tags | problem_tags, q)
        conn.close()
        return {
            "results": results,
            "calibration_note": cal_note,
            "guidance": ("Refuted/inconclusive results are ranked first on purpose — "
                         "do not silently retry a dead end. Propose a NEW hypothesis "
                         "with a committed prediction, or supersede a prior one."),
        }

    def _recall_calibration_note(self, conn, tags: set[str], q: str):
        # find a diagnostic mentioned by id in the query or associated with the tags
        diag_ids = [r["id"] for r in conn.execute("SELECT id FROM diagnostics").fetchall()]
        target = None
        for did in diag_ids:
            if did in q:
                target = did
                break
        if not target:
            return None
        rows = conn.execute(
            "SELECT matched FROM evaluation WHERE diagnostic_id=?", (target,)
        ).fetchall()
        if not rows:
            return None
        n = len(rows)
        k = sum(r["matched"] for r in rows)
        caution = " — treat with caution" if n >= 2 and k / n < 0.5 else ""
        return f"your predictions on `{target}` have been right {k}/{n} times{caution}"

    # ------------------------------------------------------------------ #
    # CAUSAL MAP (brief §5)
    # ------------------------------------------------------------------ #
    def causal_map(self, problem_id=None, tag=None) -> dict:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM intervention_effects").fetchall()
        conn.close()

        def keep(r):
            if problem_id and r["problem_id"] != problem_id:
                return False
            if tag:
                tags = set((r["topic_tags"] or "").split(",")) | set((r["problem_tags"] or "").split(","))
                if tag not in tags:
                    return False
            return True

        agg: dict[tuple[str, str], dict] = {}
        for r in rows:
            if not keep(r):
                continue
            key = (r["intervention_id"], r["diagnostic_id"])
            a = agg.setdefault(key, {"deltas": [], "up": 0, "down": 0, "flat": 0, "experiments": set()})
            a["deltas"].append(r["delta"])
            a["experiments"].add(r["experiment_id"])
            a[{"up": "up", "down": "down", "none": "flat"}[r["direction"]]] += 1

        edges = []
        for (iv, diag), a in agg.items():
            n = len(a["deltas"])
            mean = sum(a["deltas"]) / n if n else 0.0
            sign = "↑" if mean > 0 else ("↓" if mean < 0 else "→")
            edges.append({
                "intervention": iv,
                "diagnostic": diag,
                "mean_delta": round(mean, 6),
                "sign": sign,
                "n_experiments": len(a["experiments"]),
                "up": a["up"], "down": a["down"], "flat": a["flat"],
            })
        edges.sort(key=lambda e: (e["intervention"], -abs(e["mean_delta"])))

        lines = ["# Causal map",
                 f"scope: {'problem=' + problem_id if problem_id else ('tag=' + tag if tag else 'all')}",
                 ""]
        if not edges:
            lines.append("_(no closed experiments in scope yet)_")
        for e in edges:
            lines.append(f"- [[{e['intervention']}]] {e['sign']} [[{e['diagnostic']}]]  "
                         f"mean Δ {e['mean_delta']:+g} over {e['n_experiments']} exp "
                         f"(↑{e['up']} ↓{e['down']} →{e['flat']})")
        return {"scope": {"problem_id": problem_id, "tag": tag},
                "edges": edges, "text": "\n".join(lines)}

    def write_canvas(self, problem_id=None, tag=None) -> str:
        """Emit an Obsidian .canvas for the causal map (brief §5, optional)."""
        import json

        cm = self.causal_map(problem_id=problem_id, tag=tag)
        nodes, edges_json = [], []
        seen: dict[str, str] = {}
        x_iv, x_dg, y = 0, 500, 0

        def node_for(name, col):
            nonlocal y
            if name in seen:
                return seen[name]
            nid = f"n{len(seen)}"
            seen[name] = nid
            nx = x_iv if col == "iv" else x_dg
            nodes.append({"id": nid, "type": "text", "text": f"[[{name}]]",
                          "x": nx, "y": y, "width": 220, "height": 60})
            y += 90
            return nid

        for e in cm["edges"]:
            a = node_for(e["intervention"], "iv")
            b = node_for(e["diagnostic"], "dg")
            edges_json.append({"id": f"e{len(edges_json)}", "fromNode": a,
                               "toNode": b, "label": f"{e['sign']} {e['mean_delta']:+g}"})
        canvas = {"nodes": nodes, "edges": edges_json}
        scope = problem_id or tag or "all"
        path = self.vault.root / "_canvas" / f"causal-{slugify(scope)}.canvas"
        path.write_text(json.dumps(canvas, indent=2), encoding="utf-8")
        return str(path)

    # ------------------------------------------------------------------ #
    # CALIBRATION (brief §5)
    # ------------------------------------------------------------------ #
    def calibration(self, diagnostic_id=None, tag=None, intervention_id=None) -> dict:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM evaluation").fetchall()
        conn.close()

        def keep(r):
            if diagnostic_id and r["diagnostic_id"] != diagnostic_id:
                return False
            if intervention_id and intervention_id not in (r["interventions"] or "").split(","):
                return False
            if tag:
                tags = set((r["topic_tags"] or "").split(",")) | set((r["problem_tags"] or "").split(","))
                if tag not in tags:
                    return False
            return True

        kept = [r for r in rows if keep(r)]
        n = len(kept)
        k = sum(r["matched"] for r in kept)
        by_diag: dict[str, list[int]] = {}
        for r in kept:
            by_diag.setdefault(r["diagnostic_id"], []).append(r["matched"])
        per_diag = {d: {"right": sum(v), "n": len(v),
                        "accuracy": round(sum(v) / len(v), 3)} for d, v in by_diag.items()}

        scope = ("diagnostic=" + diagnostic_id if diagnostic_id else
                 "intervention=" + intervention_id if intervention_id else
                 "tag=" + tag if tag else "all")
        acc = round(k / n, 3) if n else None
        note = None
        if n >= 2 and acc is not None and acc < 0.5:
            note = "miscalibrated: your predictions here are wrong more often than right"
        return {
            "scope": scope,
            "n": n, "right": k, "accuracy": acc,
            "per_diagnostic": per_diag,
            "note": note,
        }
