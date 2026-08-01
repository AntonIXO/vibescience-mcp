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
    Tag,
    TagAxis,
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
    # Tag registry (a FIXED vocabulary, like diagnostics — brief §3/§4)
    # ------------------------------------------------------------------ #
    def register_tag(self, id: str, axis: str, description: str = "",
                     aliases: list[str] | None = None) -> Tag:
        """Register a tag on an axis. Idempotent: re-registering MERGES aliases.

        Merging rather than overwriting keeps resolution *additive* — an alias
        recorded once is never silently dropped by a later call.
        """
        tid = slugify(id)
        ax = TagAxis(axis)
        incoming = [slugify(a) for a in (aliases or [])]
        if self.vault.exists("tags", tid):
            existing = self.vault.read_tag(tid)
            if existing.axis != ax:
                raise VibeScienceError(
                    f"Tag '{tid}' is already registered on axis '{existing.axis.value}'. "
                    f"A tag lives on exactly one axis; pick a distinct id for the "
                    f"'{ax.value}' sense."
                )
            merged = list(dict.fromkeys(existing.aliases + incoming))
            t = existing.model_copy(update={
                "aliases": merged,
                "description": description or existing.description,
            })
        else:
            t = Tag(id=tid, axis=ax, description=description, aliases=incoming)

        # An alias may not collide with a real tag id or another tag's alias.
        for other in self.vault.all_tags():
            if other.id == t.id:
                continue
            clash = set(t.aliases) & ({other.id} | set(other.aliases))
            if clash:
                raise VibeScienceError(
                    f"Alias {sorted(clash)} already belongs to tag '{other.id}'. "
                    f"Aliases must resolve to exactly one canonical tag."
                )
        if t.id in set(t.aliases):
            raise VibeScienceError(f"Tag '{t.id}' cannot alias itself.")
        self.vault.write_tag(t)
        self.reindex()
        return t

    def list_tags(self, axis: str | None = None) -> list[dict]:
        """The registered vocabulary + how often each tag is actually used.

        ``n_uses``/``kinds`` are what make tag rot visible: a tag with n_uses<=1
        or a single kind is an orphan that connects nothing.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT tag, entity_kind, COUNT(*) n FROM tags GROUP BY tag, entity_kind"
        ).fetchall()
        conn.close()
        usage: dict[str, dict] = {}
        for r in rows:
            u = usage.setdefault(r["tag"], {"n_uses": 0, "kinds": {}})
            u["n_uses"] += r["n"]
            u["kinds"][r["entity_kind"]] = r["n"]

        out = []
        for t in self.vault.all_tags():
            if axis and t.axis.value != axis:
                continue
            u = usage.get(t.id, {"n_uses": 0, "kinds": {}})
            out.append({
                "id": t.id, "axis": t.axis.value, "description": t.description,
                "aliases": t.aliases, "n_uses": u["n_uses"], "kinds": u["kinds"],
                "orphan": u["n_uses"] <= 1 or len(u["kinds"]) <= 1,
            })
        out.sort(key=lambda x: (-x["n_uses"], x["id"]))
        return out

    def _tag_maps(self) -> tuple[dict[str, Tag], dict[str, str]]:
        """(canonical id -> Tag, any surface form -> canonical id)."""
        canon: dict[str, Tag] = {}
        alias: dict[str, str] = {}
        for t in self.vault.all_tags():
            canon[t.id] = t
            alias[t.id] = t.id
            for a in t.aliases:
                alias[a] = t.id
        return canon, alias

    @staticmethod
    def _near(needle: str, hay: list[str], n: int = 3) -> list[str]:
        import difflib
        return difflib.get_close_matches(needle, hay, n=n, cutoff=0.6)

    def resolve_tags(self, tags: list[str] | None, axis: str,
                     strict: bool = True) -> list[str]:
        """Map surface forms to canonical registered tag ids on one axis.

        This is the whole point of the registry: free text becomes a controlled
        vocabulary at the write boundary, so ``recall`` can never miss because a
        past session wrote ``causality`` and this one wrote ``causal-inference``.
        With ``strict`` an unknown tag is REJECTED with near-match suggestions
        rather than silently creating an orphan.
        """
        if not tags:
            return []
        canon, alias = self._tag_maps()
        ax = TagAxis(axis)
        out: list[str] = []
        for raw in tags:
            key = slugify(raw)
            cid = alias.get(key)
            if cid is None:
                if not strict:
                    continue
                pool = [t for t, tg in canon.items() if tg.axis == ax]
                sug = self._near(key, pool + [a for a in alias if a not in canon])
                hint = f" Did you mean {sug}?" if sug else ""
                raise VibeScienceError(
                    f"Unknown {ax.value} tag '{raw}'. Tags are a REGISTERED "
                    f"vocabulary (like diagnostics) so recall cannot miss on a "
                    f"synonym.{hint} Call list_tags(axis='{ax.value}') to see the "
                    f"vocabulary, then register_tag(id='{key}', axis='{ax.value}') "
                    f"if it is genuinely new — or add it as an alias of an "
                    f"existing tag."
                )
            if canon[cid].axis != ax:
                raise VibeScienceError(
                    f"Tag '{cid}' is registered on axis '{canon[cid].axis.value}', "
                    f"not '{ax.value}'. topic_tags describe subject matter; "
                    f"problem_tags describe the failure mode."
                )
            if cid not in out:
                out.append(cid)
        return out

    def _inherit_tags(self, problem_id: str, own: list[str] | None,
                      interventions: list[str] | None, axis: str) -> list[str]:
        """Resolve a hypothesis's own tags, then INHERIT from its context.

        Tag rot came from every entity being tagged by hand and drifting apart.
        A hypothesis is *about* its problem and *applies* its interventions, so
        it inherits their tags automatically. This is what makes a tag a real
        edge (problem ↔ hypothesis ↔ intervention ↔ paper) instead of a label
        that lives on exactly one node.
        """
        out = self.resolve_tags(own, axis)
        if self.vault.exists("problems", problem_id):
            p = self.vault.read_problem(problem_id)
            inherited = p.topic_tags if axis == "topic" else p.problem_tags
            for t in inherited:
                if t not in out:
                    out.append(t)
        if axis == "topic":
            for iv in interventions or []:
                if self.vault.exists("interventions", iv):
                    for t in self.vault.read_intervention(iv).topic_tags:
                        if t not in out:
                            out.append(t)
        return out

    # ------------------------------------------------------------------ #
    # Registry: diagnostics + interventions + papers
    # ------------------------------------------------------------------ #
    def register_diagnostic(self, name, unit="", direction="neutral",
                            description="", topic_tags=None, id=None) -> Diagnostic:
        d = Diagnostic(id=id or slugify(name), name=name, unit=unit,
                       direction=DiagDirection(direction), description=description,
                       topic_tags=self.resolve_tags(topic_tags, "topic"))
        self.vault.write_diagnostic(d)
        self.reindex()
        return d

    def list_diagnostics(self) -> list[Diagnostic]:
        return list(self.vault.all_diagnostics())

    def _diag_ids(self) -> set[str]:
        return set(self.vault.ids("diagnostics"))

    def register_intervention(self, name, description="", topic_tags=None, id=None) -> Intervention:
        iv = Intervention(id=id or slugify(name), name=name, description=description,
                          topic_tags=self.resolve_tags(topic_tags, "topic"))
        self.vault.write_intervention(iv)
        self.reindex()
        return iv

    def list_interventions(self) -> list[Intervention]:
        return list(self.vault.all_interventions())

    def add_paper(self, title, arxiv_id_or_url="", key_claims=None, topic_tags=None, id=None) -> Paper:
        p = Paper(id=id or slugify(title), title=title, arxiv_id_or_url=arxiv_id_or_url,
                  key_claims=key_claims or [],
                  topic_tags=self.resolve_tags(topic_tags, "topic"))
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
                    topic_tags=self.resolve_tags(topic_tags, "topic"),
                    problem_tags=self.resolve_tags(problem_tags, "problem"),
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
            plan=plan,
            topic_tags=self._inherit_tags(problem_id, topic_tags, interventions, "topic"),
            problem_tags=self._inherit_tags(problem_id, problem_tags, None, "problem"),
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
                         config_note="", parent_experiment_id=None, id=None) -> Experiment:
        if not self.vault.exists("hypotheses", hypothesis_id):
            raise VibeScienceError(f"Unknown hypothesis '{hypothesis_id}'.")
        if parent_experiment_id and not self.vault.exists("experiments", parent_experiment_id):
            raise VibeScienceError(
                f"Unknown parent_experiment_id '{parent_experiment_id}'. "
                f"The parent must be an experiment you already ran — that link is "
                f"what makes lineage()/children()/leaves() work."
            )
        if not git_ref:
            git_ref = self._git_ref()
        eid = id or slugify(f"{hypothesis_id}-exp-{now_iso()[:19]}")
        e = Experiment(id=eid, hypothesis_id=hypothesis_id, git_ref=git_ref or "",
                       external_run=external_run, config_note=config_note,
                       parent_experiment_id=parent_experiment_id)
        h = self.vault.read_hypothesis(hypothesis_id)
        self.vault.write_experiment(e, tags=h.topic_tags + h.problem_tags)
        # move the hypothesis into 'testing'
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
        self.vault.write_experiment(e, predicted=pred, tags=h.topic_tags + h.problem_tags)
        self.reindex()
        return e

    def abort_experiment(self, experiment_id, crash_reason: str, notes="") -> dict:
        """Close a run that CRASHED — no measurement, therefore no evidence.

        Karpathy's ``results.tsv`` has three statuses (keep / discard / crash);
        we only had two. Without this a crashed run either stays open forever
        (invisible to lineage and to the frontier) or gets faked as a null
        result, which poisons calibration with a prediction that was never
        actually tested. ``crashed`` is bookkeeping, not evidence: it never
        enters the ``evaluation`` table and the hypothesis stays retryable.
        """
        if not self.vault.exists("experiments", experiment_id):
            raise VibeScienceError(f"Unknown experiment '{experiment_id}'.")
        if not crash_reason:
            raise VibeScienceError(
                "abort_experiment requires a crash_reason (the error/why it died)."
            )
        e = self.vault.read_experiment(experiment_id)
        if e.closed:
            raise VibeScienceError(f"Experiment '{experiment_id}' is already closed.")
        e.verdict = Verdict.crashed
        e.crash_reason = crash_reason
        e.closed = True
        if notes:
            e.notes = notes
        h = self.vault.read_hypothesis(e.hypothesis_id)
        self.vault.write_experiment(e, tags=h.topic_tags + h.problem_tags)
        # hypothesis stays 'testing' — a crash is not a refutation
        h.status = verdict_to_hypothesis_status(e.verdict)
        self.vault.write_hypothesis(h)
        self.reindex()
        return {
            "experiment_id": e.id,
            "verdict": e.verdict.value,
            "crash_reason": crash_reason,
            "hypothesis_status": h.status.value,
            "suggested_next_action": (
                "crash recorded, NOT counted as evidence — fix the run and start a "
                f"new experiment with parent_experiment_id='{e.id}', or supersede "
                "the hypothesis if the idea itself is broken"
            ),
        }

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
        self.vault.write_experiment(e, predicted=pred, tags=h.topic_tags + h.problem_tags)
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
    # Experiment DAG (AgentHub's `ah children` / `ah leaves` / `ah lineage`)
    # ------------------------------------------------------------------ #
    def _exp_rows(self) -> list[dict]:
        conn = self._conn()
        rows = [dict(r) for r in conn.execute(
            "SELECT id, parent_experiment_id, hypothesis_id, verdict, closed, "
            "created_at FROM experiments"
        ).fetchall()]
        conn.close()
        return rows

    def lineage(self, experiment_id: str) -> dict:
        """Ancestry path root → this experiment (AgentHub ``ah lineage``).

        The verdict chain along the path is the actual research narrative:
        which prediction failed where, and what was tried on top of it.
        """
        rows = {r["id"]: r for r in self._exp_rows()}
        if experiment_id not in rows:
            raise VibeScienceError(f"Unknown experiment '{experiment_id}'.")
        path, seen, cur = [], set(), experiment_id
        while cur and cur in rows and cur not in seen:
            seen.add(cur)
            r = rows[cur]
            path.append({"experiment_id": r["id"], "hypothesis_id": r["hypothesis_id"],
                         "verdict": r["verdict"], "closed": bool(r["closed"])})
            cur = r["parent_experiment_id"]
        path.reverse()
        return {"experiment_id": experiment_id, "depth": len(path), "path": path}

    def children(self, experiment_id: str) -> dict:
        """What was tried ON TOP of this result (AgentHub ``ah children``)."""
        rows = self._exp_rows()
        if experiment_id not in {r["id"] for r in rows}:
            raise VibeScienceError(f"Unknown experiment '{experiment_id}'.")
        kids = [{"experiment_id": r["id"], "hypothesis_id": r["hypothesis_id"],
                 "verdict": r["verdict"], "closed": bool(r["closed"])}
                for r in rows if r["parent_experiment_id"] == experiment_id]
        return {"experiment_id": experiment_id, "n_children": len(kids), "children": kids}

    def leaves(self, unevaluated_only: bool = False) -> dict:
        """The research frontier: experiments nothing was built on top of.

        ``unevaluated_only`` narrows to leaves that are still open — the dangling
        work. That is the question a flat experiment list cannot answer.
        """
        rows = self._exp_rows()
        parents = {r["parent_experiment_id"] for r in rows if r["parent_experiment_id"]}
        out = []
        for r in rows:
            if r["id"] in parents:
                continue
            if unevaluated_only and r["closed"]:
                continue
            out.append({"experiment_id": r["id"], "hypothesis_id": r["hypothesis_id"],
                        "verdict": r["verdict"], "closed": bool(r["closed"]),
                        "created_at": r["created_at"]})
        out.sort(key=lambda x: x["created_at"], reverse=True)
        return {"n_leaves": len(out), "unevaluated_only": unevaluated_only, "leaves": out}

    # ------------------------------------------------------------------ #
    # RECALL — negative-first pre-mortem gate (brief §6, §8)
    # ------------------------------------------------------------------ #
    def recall(self, query=None, topic_tags=None, problem_tags=None,
               problem_id=None, limit=10) -> dict:
        """Negative-first pre-mortem search across the WHOLE vault.

        Hypotheses stay primary (they carry verdicts), but papers, interventions
        and problems are searched too: 69% of tag mass lived on those kinds and
        was unreachable when recall only queried ``hypotheses`` — a tag that
        matched two papers returned nothing and the agent concluded "no prior
        work". Tags are resolved through the registry first, so a synonym still
        hits.
        """
        conn = self._conn()
        q = (query or "").lower()
        topic_tags = set(self.resolve_tags(list(topic_tags or []), "topic", strict=False))
        problem_tags = set(self.resolve_tags(list(problem_tags or []), "problem", strict=False))

        rows = conn.execute(
            "SELECT h.id, h.problem_id, h.statement, h.rationale, h.status "
            "FROM hypotheses h"
        ).fetchall()

        def tags_for(eid, kind="hypothesis"):
            tt = conn.execute(
                "SELECT tag, axis FROM tags WHERE entity_id=? AND entity_kind=?",
                (eid, kind),
            ).fetchall()
            topic = {r["tag"] for r in tt if r["axis"] == "topic"}
            prob = {r["tag"] for r in tt if r["axis"] == "problem"}
            return topic, prob

        def kw_score(hay: str):
            """Shared keyword scoring: exact phrase, else per-word hits."""
            if not q:
                return 0.0, None
            hay = hay.lower()
            if q in hay:
                return 1.5, "keyword"
            words = [w for w in q.split() if len(w) > 2]
            hits = sum(1 for w in words if w in hay)
            return (0.5 * hits, f"keyword×{hits}") if hits else (0.0, None)

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
            ks, kr = kw_score(r["statement"] + " " + (r["rationale"] or ""))
            if kr:
                score += ks; reasons.append(kr)
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
                if ev and ev["observed"] is not None and ev["delta"] is not None:
                    why = (f"{status}: predicted {ev['diagnostic_id']} {ev['predicted']}, "
                           f"observed {ev['observed']} (Δ {ev['delta']:+g})")
                elif ev:
                    # The primary diagnostic was PREDICTED but never measured, so
                    # observed/delta are NULL. Formatting NULL used to raise
                    # TypeError and take the whole pre-mortem gate down — and it
                    # only ever triggered on inconclusive/refuted rows, i.e. the
                    # negative results this tool exists to surface.
                    why = (f"{status}: predicted {ev['diagnostic_id']} "
                           f"{ev['predicted']}, but that diagnostic was never "
                           f"measured — re-run and record it")
                else:
                    why = f"{status}: no confirming measurement"

            scored.append({
                "kind": "hypothesis",
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

        # --- context layer: the kinds recall used to be blind to -------------
        # Scored lower than hypotheses on purpose: grounding, not verdicts.
        context = []
        ctx_specs = [
            ("paper", "papers", "SELECT id, title AS label, arxiv_id_or_url AS extra FROM papers"),
            ("intervention", "interventions",
             "SELECT id, name AS label, description AS extra FROM interventions"),
            ("problem", "problems", "SELECT id, title AS label, description AS extra FROM problems"),
        ]
        for kind, _tbl, sql in ctx_specs:
            for r in conn.execute(sql).fetchall():
                topic, prob = tags_for(r["id"], kind)
                score, reasons = 0.0, []
                tt_hit = topic_tags & topic
                pt_hit = problem_tags & prob
                if tt_hit:
                    score += 1.5 * len(tt_hit); reasons.append("topic:" + ",".join(sorted(tt_hit)))
                if pt_hit:
                    score += 1.5 * len(pt_hit); reasons.append("problem:" + ",".join(sorted(pt_hit)))
                if problem_id and kind == "problem" and r["id"] == problem_id:
                    score += 2.0; reasons.append("same problem")
                ks, kr = kw_score(f"{r['label']} {r['extra'] or ''}")
                if kr:
                    score += ks; reasons.append(kr)
                if score <= 0:
                    continue
                context.append({"kind": kind, "id": r["id"], "label": r["label"],
                                "score": round(score, 2), "match_reasons": reasons})
        context.sort(key=lambda x: x["score"], reverse=True)
        context = context[:limit]

        # calibration note if the query touches a tracked diagnostic/tag
        cal_note = self._recall_calibration_note(conn, topic_tags | problem_tags, q)
        conn.close()
        return {
            "results": results,
            "context": context,
            "calibration_note": cal_note,
            "guidance": ("Refuted/inconclusive results are ranked first on purpose — "
                         "do not silently retry a dead end. Propose a NEW hypothesis "
                         "with a committed prediction, or supersede a prior one. "
                         "`context` holds related papers/interventions/problems — "
                         "grounding for the new hypothesis, not evidence."),
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
