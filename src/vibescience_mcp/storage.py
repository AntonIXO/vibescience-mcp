"""Markdown vault I/O — the source of truth (brief §2.1, §7).

Every entity is one ``.md`` file: YAML frontmatter carries structured fields,
the body carries prose (statement / rationale / plan / notes / description) plus
``[[wikilinks]]`` so Obsidian renders the graph for free. Reads reconstruct the
structured fields from frontmatter and the prose from headed body sections, so
the round-trip is lossless.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import frontmatter

from .models import (
    Diagnostic,
    Direction,
    Experiment,
    Hypothesis,
    Intervention,
    Paper,
    Problem,
)

KINDS = {
    "problems": Problem,
    "hypotheses": Hypothesis,
    "experiments": Experiment,
    "diagnostics": Diagnostic,
    "interventions": Intervention,
    "papers": Paper,
}
SUBDIRS = list(KINDS) + ["_canvas"]

_section_re = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse_sections(body: str) -> dict[str, str]:
    """Split a markdown body into ``{lower_header: text}`` blocks."""
    out: dict[str, str] = {}
    matches = list(_section_re.finditer(body))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out[m.group(1).strip().lower()] = body[start:end].strip()
    return out


def wl(target: str) -> str:
    return f"[[{target}]]"


class Vault:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        for d in SUBDIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------- #
    def path(self, kind: str, id_: str) -> Path:
        return self.root / kind / f"{id_}.md"

    def exists(self, kind: str, id_: str) -> bool:
        return self.path(kind, id_).exists()

    def ids(self, kind: str) -> list[str]:
        return sorted(p.stem for p in (self.root / kind).glob("*.md"))

    # -- generic write ----------------------------------------------------- #
    def _write(self, kind: str, id_: str, meta: dict, body: str) -> Path:
        post = frontmatter.Post(body.strip() + "\n", **meta)
        p = self.path(kind, id_)
        p.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")
        return p

    def _load(self, kind: str, id_: str) -> tuple[dict, dict]:
        post = frontmatter.load(self.path(kind, id_))
        return dict(post.metadata), parse_sections(post.content)

    # ===================================================================== #
    # Problem
    # ===================================================================== #
    def write_problem(self, p: Problem) -> Path:
        meta = p.model_dump(mode="json", exclude={"description"})
        tags = " ".join(f"#{t}" for t in p.topic_tags + p.problem_tags)
        body = f"## Description\n{p.description}\n"
        if p.paper_refs:
            body += "\n## Links\nCites " + ", ".join(wl(r) for r in p.paper_refs) + ".\n"
        if tags:
            body += f"\n{tags}\n"
        return self._write("problems", p.id, meta, body)

    def read_problem(self, id_: str) -> Problem:
        meta, sec = self._load("problems", id_)
        meta["description"] = sec.get("description", "")
        return Problem.model_validate(meta)

    # ===================================================================== #
    # Hypothesis
    # ===================================================================== #
    def write_hypothesis(self, h: Hypothesis) -> Path:
        meta = h.model_dump(mode="json", exclude={"statement", "rationale", "plan"})
        body = f"## Statement\n{h.statement}\n\n## Rationale\n{h.rationale}\n\n## Plan\n{h.plan}\n"
        # Links block for the graph
        links: list[str] = [f"Tests problem {wl(h.problem_id)}."]
        if h.interventions:
            links.append("Applies " + ", ".join(wl(i) for i in h.interventions) + ".")
        for pe in h.predicted_effects:
            links.append(f"Predicts {wl(pe.diagnostic_id)} ({pe.direction.value}).")
        if h.paper_refs:
            links.append("Cites " + ", ".join(wl(r) for r in h.paper_refs) + ".")
        if h.supersedes:
            links.append(f"Supersedes {wl(h.supersedes)}.")
        body += "\n## Links\n" + "\n".join(links) + "\n"
        tags = " ".join(f"#{t}" for t in h.topic_tags + h.problem_tags)
        if tags:
            body += f"\n{tags}\n"
        return self._write("hypotheses", h.id, meta, body)

    def read_hypothesis(self, id_: str) -> Hypothesis:
        meta, sec = self._load("hypotheses", id_)
        meta["statement"] = sec.get("statement", "")
        meta["rationale"] = sec.get("rationale", "")
        meta["plan"] = sec.get("plan", "")
        return Hypothesis.model_validate(meta)

    # ===================================================================== #
    # Experiment
    # ===================================================================== #
    def write_experiment(self, e: Experiment, diags: dict[str, Diagnostic] | None = None,
                         predicted: dict[str, Direction] | None = None) -> Path:
        meta = e.model_dump(mode="json", exclude={"notes"})
        body = f"## Summary\nTests hypothesis {wl(e.hypothesis_id)}.\n"
        if e.git_ref:
            body += f"At commit `{e.git_ref}`.\n"
        if e.external_run:
            body += f"External run: {e.external_run}\n"
        # Measurements block, Obsidian-friendly (mirrors brief §7 example)
        if e.diagnostics_measured:
            body += "\n## Measurements\n"
            obs = {o.diagnostic_id: o for o in e.observed_effects}
            for m in e.diagnostics_measured:
                o = obs.get(m.diagnostic_id)
                line = f"- {wl(m.diagnostic_id)}: before {m.before} → after {m.after}"
                if o is not None:
                    arrow = {"up": "up", "down": "down", "none": "flat"}[o.direction.value]
                    line += f"  (Δ {o.delta:+g}, {arrow}"
                    if predicted and m.diagnostic_id in predicted:
                        pd = predicted[m.diagnostic_id].value
                        ok = "✓" if o.direction.value == pd else "✗"
                        line += f" {ok} predicted {pd}"
                    line += ")"
                body += line + "\n"
        if e.verdict:
            body += f"\n**Verdict:** {e.verdict.value}\n"
        if e.notes:
            body += f"\n## Notes\n{e.notes}\n"
        return self._write("experiments", e.id, meta, body)

    def read_experiment(self, id_: str) -> Experiment:
        meta, sec = self._load("experiments", id_)
        meta["notes"] = sec.get("notes", "")
        return Experiment.model_validate(meta)

    # ===================================================================== #
    # Registry entities (all-frontmatter, small body for Obsidian)
    # ===================================================================== #
    def write_diagnostic(self, d: Diagnostic) -> Path:
        meta = d.model_dump(mode="json", exclude={"description"})
        body = f"## Description\n{d.description}\n\n*Unit:* `{d.unit}` — *{d.direction.value}*\n"
        return self._write("diagnostics", d.id, meta, body)

    def read_diagnostic(self, id_: str) -> Diagnostic:
        meta, sec = self._load("diagnostics", id_)
        meta["description"] = sec.get("description", "")
        return Diagnostic.model_validate(meta)

    def write_intervention(self, i: Intervention) -> Path:
        meta = i.model_dump(mode="json", exclude={"description"})
        body = f"## Description\n{i.description}\n"
        tags = " ".join(f"#{t}" for t in i.topic_tags)
        if tags:
            body += f"\n{tags}\n"
        return self._write("interventions", i.id, meta, body)

    def read_intervention(self, id_: str) -> Intervention:
        meta, sec = self._load("interventions", id_)
        meta["description"] = sec.get("description", "")
        return Intervention.model_validate(meta)

    def write_paper(self, p: Paper) -> Path:
        meta = p.model_dump(mode="json", exclude={"key_claims"})
        body = "## Key claims\n" + "\n".join(f"- {c}" for c in p.key_claims) + "\n"
        tags = " ".join(f"#{t}" for t in p.topic_tags)
        if tags:
            body += f"\n{tags}\n"
        return self._write("papers", p.id, meta, body)

    def read_paper(self, id_: str) -> Paper:
        meta, sec = self._load("papers", id_)
        claims = [
            ln[2:].strip()
            for ln in sec.get("key claims", "").splitlines()
            if ln.strip().startswith("- ")
        ]
        meta["key_claims"] = claims
        return Paper.model_validate(meta)

    # -- iteration helpers ------------------------------------------------- #
    def all_problems(self) -> Iterable[Problem]:
        return [self.read_problem(i) for i in self.ids("problems")]

    def all_hypotheses(self) -> Iterable[Hypothesis]:
        return [self.read_hypothesis(i) for i in self.ids("hypotheses")]

    def all_experiments(self) -> Iterable[Experiment]:
        return [self.read_experiment(i) for i in self.ids("experiments")]

    def all_diagnostics(self) -> Iterable[Diagnostic]:
        return [self.read_diagnostic(i) for i in self.ids("diagnostics")]

    def all_interventions(self) -> Iterable[Intervention]:
        return [self.read_intervention(i) for i in self.ids("interventions")]

    def all_papers(self) -> Iterable[Paper]:
        return [self.read_paper(i) for i in self.ids("papers")]
