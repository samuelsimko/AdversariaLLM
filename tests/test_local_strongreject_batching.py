import judges


def test_judge_classifier_sequence_uses_batched_strongreject(monkeypatch):
    calls = []

    def fake_batch(prompts, responses):
        calls.append((prompts, responses))
        return [0.25, 0.75]

    def fail_single(*_args, **_kwargs):
        raise AssertionError("per-example StrongREJECT path should not be used")

    monkeypatch.setattr(judges, "judge_strongreject_batch", fake_batch)
    monkeypatch.setattr(judges, "get_strongreject_score", fail_single)

    results = judges.judge_classifier_sequence(
        "strongreject",
        ["prompt 1", "prompt 2"],
        ["response 1", "response 2"],
    )

    assert calls == [(["prompt 1", "prompt 2"], ["response 1", "response 2"])]
    assert results == [{"strongreject": 0.25}, {"strongreject": 0.75}]
