"""Ingest the run-report sidecars optiHealth-EiV already writes.

Every EiV run emits a report.json carrying implementation_sha256 (14 files),
config/split/protocol hashes, git sha, exact device, peak VRAM and oom/nonfinite
flags. The vault stored NONE of it — only a prose config_note — even though
`curate-long-running-experiments` §3/§7 require freezing and independently
verifying exactly those fields. Retyping hashes into prose loses them to search
and to audit.

The fixture below mirrors the real schema, verified on the GPU host at
/root/optiHealth-EiV/outputs/experiments/*/runs/*/*/seed_*/report.json.
"""

import json

import pytest

from vibescience_mcp.core import Store, VibeScienceError

REPORT = {
    "arm": "dcl",
    "objective": "dcl",
    "oom": False,
    "nonfinite": False,
    "config_sha256": "9910a9a939e36599ce1d080b73f7e2876ee67b75620200226d9eb0f0ee2e",
    "split_sha256": "43dcafef4a0a4a0e4cae3c6bca0b067599ae61b35d1ea1983e4a16ab4ac2",
    "protocol_sha256": "93fa06a385bf9bf6b2f2381b941cd44c04f27baf00cb4f8405a277c60e29",
    "corrected_mask_sha256": "31f0620e135192349a0add2756d624d0b0e1fa5a519bc776cb8b9e288964",
    "implementation_digest": "dc45212c44f09693b9d03b1c838c4ceb4cd33409ad0588f28f8c06ac6f86",
    "environment": {
        "git_sha": "955c1b5d19b470ee2b88c36fb765d69dcf3f852d",
        "cuda_device": "NVIDIA GeForce RTX 5070",
        "torch": "2.11.0+cu130",
        "hostname": "cachyos-x8664",
        "python": "3.11.15",
    },
    "gpu": {"peak_memory_allocated_gib": 10.305453300476074},
}

OOM_REPORT = dict(
    REPORT, oom=True,
    error={"type": "OutOfMemoryError",
           "message": "CUDA out of memory. Tried to allocate 100.00 MiB."},
)


def _exp(vault, report):
    s = Store(vault)
    s.register_diagnostic("m", direction="higher_better", id="m")
    s.create_problem("p", id="p")
    s.propose_hypothesis("p", "m up", id="h",
                         predicted_effects=[{"diagnostic_id": "m", "direction": "up"}])
    s.start_experiment("h", id="e")
    path = vault / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return s, str(path)


def test_import_captures_integrity_hashes(vault):
    s, path = _exp(vault, REPORT)
    out = s.import_run_report("e", path)
    prov = s.vault.read_experiment("e").provenance

    assert prov["config_sha256"].startswith("9910a9a9")
    assert prov["split_sha256"].startswith("43dcafef")
    assert prov["protocol_sha256"].startswith("93fa06a3")
    assert prov["implementation_digest"].startswith("dc45212c")
    assert prov["git_sha"].startswith("955c1b5d")
    assert prov["cuda_device"] == "NVIDIA GeForce RTX 5070"
    assert prov["peak_memory_allocated_gib"] == "10.305453300476074"
    assert prov["arm"] == "dcl"
    assert out["crash_detected"] is False


def test_import_flags_oom_and_refuses_to_invent_a_result(vault):
    s, path = _exp(vault, OOM_REPORT)
    out = s.import_run_report("e", path)
    assert out["crash_detected"] is True
    assert "OutOfMemoryError" in out["crash_reason"]
    assert "abort_experiment" in out["suggested_next_action"]


def test_import_flags_nonfinite_loss(vault):
    s, path = _exp(vault, dict(REPORT, nonfinite=True))
    assert s.import_run_report("e", path)["crash_detected"] is True


def test_import_is_idempotent_and_merges(vault):
    s, path = _exp(vault, REPORT)
    s.import_run_report("e", path)
    s.import_run_report("e", path)
    prov = s.vault.read_experiment("e").provenance
    assert prov["git_sha"].startswith("955c1b5d")


def test_provenance_survives_reindex(vault):
    s, path = _exp(vault, REPORT)
    s.import_run_report("e", path)
    before = s.vault.read_experiment("e").provenance
    s.db_path.unlink()
    s.reindex()
    assert s.vault.read_experiment("e").provenance == before


def test_import_rejects_a_missing_report(vault):
    s, _ = _exp(vault, REPORT)
    with pytest.raises(VibeScienceError, match="not found"):
        s.import_run_report("e", str(vault / "nope.json"))


def test_import_rejects_malformed_json(vault):
    s, _ = _exp(vault, REPORT)
    bad = vault / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(VibeScienceError, match="not valid JSON"):
        s.import_run_report("e", str(bad))


def test_import_rejects_unknown_experiment(vault):
    s, path = _exp(vault, REPORT)
    with pytest.raises(VibeScienceError, match="Unknown experiment"):
        s.import_run_report("nope", path)
