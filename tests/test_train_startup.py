import json

from trajsig.cache import ArtifactStore
from trajsig.config import load_config
from trajsig import train


def test_fresh_training_skips_missing_run_probes(tmp_path, monkeypatch, capsys):
    config = load_config("configs/smoke.yaml")
    seen = []

    def unexpected_probe(self, run_id, epochs):
        raise AssertionError("Fresh experiment should not probe 30 absent Drive files")

    def fake_train_one(*args, **kwargs):
        seen.append(kwargs["known_complete"])
        return f"run-{len(seen)}"

    monkeypatch.setattr(ArtifactStore, "is_complete", unexpected_probe)
    monkeypatch.setattr(train, "train_one", fake_train_one)

    identifiers = train.train_all(config, tmp_path)

    assert len(identifiers) == len(config["conditions"]) * len(config["experiment"]["seed_ids"])
    assert seen == [False] * len(identifiers)
    assert "CHECKING saved runs" in capsys.readouterr().out
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "training_complete"
