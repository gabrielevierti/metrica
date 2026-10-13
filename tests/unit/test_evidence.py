"""Evidence integrity: hashing, hash-chained audit log, case intake, annotations."""
from __future__ import annotations

import json
import os
import stat

import pytest

from metrica.evidence.audit import AuditLog, verify_chain
from metrica.evidence.case import Case, list_cases
from metrica.evidence.hashing import sha256_bytes, sha256_file, verify_file
from metrica.evidence.manifest import INTEGRITY_STATEMENT


def test_sha256_known_vector(tmp_path):
    p = tmp_path / "abc.bin"
    p.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(p) == expected
    assert sha256_bytes(b"abc") == expected
    assert verify_file(p, expected.upper())


def test_audit_chain_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.append("test.event", actor="tester", target=f"t{i}", details={"i": i})
    ok = verify_chain(log.entries())
    assert ok["valid"] and ok["entries"] == 5 and len(ok["head"]) == 64

    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    # edit one entry's details
    e = json.loads(lines[2])
    e["details"]["i"] = 99
    edited = lines[:2] + [json.dumps(e)] + lines[3:]
    assert verify_chain([json.loads(x) for x in edited]) == {"valid": False, "entries": 5, "first_broken_seq": 3,
                                                             "head": None}
    # delete an entry
    deleted = [json.loads(x) for x in lines[:1] + lines[2:]]
    assert not verify_chain(deleted)["valid"]
    # reorder
    reordered = [json.loads(x) for x in [lines[1], lines[0]] + lines[2:]]
    assert not verify_chain(reordered)["valid"]


def test_case_intake_copies_verifies_and_protects(tmp_path, tiny_video):
    case = Case.create(tmp_path / "case", "Intake test", "REF-1")
    original_hash = sha256_file(tiny_video)
    rec = case.add_video(tiny_video, copy=True, actor="tester")
    assert rec["id"] == "v1"
    assert rec["sha256_intake"] == original_hash == rec["sha256_stored"]
    stored = case.video_path("v1")
    assert stored.exists() and stored != tiny_video
    # the stored copy is read-only and the original is untouched
    assert not (stored.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    assert sha256_file(tiny_video) == original_hash
    # duplicate intake is rejected
    with pytest.raises(ValueError):
        case.add_video(tiny_video, copy=True)
    res = case.verify_originals()
    assert res[0]["ok"]
    # simulated tampering with the stored copy is detected
    os.chmod(stored, stat.S_IRUSR | stat.S_IWUSR)
    with open(stored, "r+b") as f:
        f.seek(100)
        f.write(b"\x00\x01\x02")
    assert not case.verify_originals()[0]["ok"]
    acts = [e["action"] for e in case.audit.entries()]
    assert acts[:2] == ["case.created", "evidence.video_added"]
    assert verify_chain(case.audit.entries())["valid"]


def test_reference_in_place(tmp_path, tiny_video):
    case = Case.create(tmp_path / "case", "In place")
    rec = case.add_video(tiny_video, copy=False)
    assert not rec["copied"] and case.video_path(rec["id"]) == tiny_video.resolve()
    assert not list((tmp_path / "case" / "evidence" / "originals").iterdir())


def test_annotations_are_audited(tmp_path):
    case = Case.create(tmp_path / "case", "Annotations")
    case.save_annotations("notes", [{"id": "N1", "text": "hello"}], actor="alice", action="annotation.note_added")
    assert case.annotations("notes")[0]["text"] == "hello"
    assert case.annotations("calibration") == {}
    last = case.audit.entries()[-1]
    assert last["actor"] == "alice" and last["action"] == "annotation.note_added"
    with pytest.raises(ValueError):
        case.annotations("not_a_kind")


def test_list_cases_and_existing_case(tmp_path):
    Case.create(tmp_path / "a", "A")
    Case.create(tmp_path / "b", "B")
    titles = sorted(c["title"] for c in list_cases(tmp_path))
    assert titles == ["A", "B"]
    with pytest.raises(FileExistsError):
        Case.create(tmp_path / "a", "again")


def test_integrity_statement_does_not_claim_authenticity():
    s = INTEGRITY_STATEMENT.lower()
    assert "authentic" in s and "not" in s


def test_pipeline_refuses_to_run_on_modified_evidence(tmp_path, tiny_video):
    from metrica.config import AnalysisConfig
    from metrica.pipeline.runner import run_pipeline

    case = Case.create(tmp_path / "case", "Refuse")
    case.add_video(tiny_video, copy=False)
    with open(tiny_video, "ab") as f:  # the referenced original changes after intake
        f.write(b"\x00")
    st = run_pipeline(case, AnalysisConfig.from_dict({"profile": "fast"}), actor="pytest")
    assert st["state"] == "failed" and "verification failed" in st["message"]
    assert case.audit.entries()[-1]["action"] == "analysis.refused"
