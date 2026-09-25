from copy import deepcopy

from trajsig.config import derive_seeds, run_id


def test_seeds_are_separate_and_run_id_opaque():
    seeds = derive_seeds(3)
    assert len(set(seeds.values())) == len(seeds)
    config = {
        "experiment": {"name": "test", "phase": "smoke", "seed_ids": [0, 1], "deterministic": True},
        "data": {"kind": "synthetic"},
        "training": {"epochs": 2},
        "conditions": [{"id": "C0", "description": "x"}],
        "telemetry": {},
        "analysis": {},
    }
    identifier = run_id(config, "C0", 0)
    assert len(identifier) == 20
    assert "C0" not in identifier


def test_analysis_changes_do_not_invalidate_training_run_id():
    config = {
        "experiment": {"name": "x", "phase": "smoke", "seed_ids": [0, 1], "deterministic": True},
        "data": {"kind": "synthetic"},
        "training": {"epochs": 2},
        "conditions": [{"id": "C0", "description": "x"}],
        "telemetry": {"stats": ["x"]},
        "analysis": {"signature_depths": [1]},
    }
    changed = deepcopy(config)
    changed["analysis"] = {"signature_depths": [1, 2, 3, 4], "classifier_c": 99}
    changed["experiment"]["seed_ids"] = [0, 1, 2, 3, 4]
    assert run_id(config, "C0", 0) == run_id(changed, "C0", 0)
