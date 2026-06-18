import torch


def _make_x(batch=2, seq_len=63):
    return torch.randn(batch, seq_len, 5)


def test_kronos_output_shape():
    from training.kronos_model import KronosClassifier

    model = KronosClassifier(hidden_dim=32, n_heads=2, n_layers=1)
    x = _make_x(batch=2, seq_len=63)
    logits = model(x)
    assert logits.shape == torch.Size([2, 3])


def test_kronos_freeze_unfreeze():
    from training.kronos_model import KronosClassifier

    model = KronosClassifier(hidden_dim=32, n_heads=2, n_layers=1)
    model.freeze_encoder()
    frozen = sum(1 for p in model.transformer.parameters() if not p.requires_grad)
    assert frozen > 0
    model.unfreeze_encoder()
    unfrozen = sum(1 for p in model.transformer.parameters() if p.requires_grad)
    assert unfrozen > 0


def test_kronos_predict_format():
    from training.kronos_model import KronosClassifier

    model = KronosClassifier(hidden_dim=32, n_heads=2, n_layers=1)
    x = [[0.001, 0.002, -0.001, 0.0015, 0.5]] * 63  # 63 bars, 5 features each
    result = model.predict(x)
    assert set(result.keys()) == {"long_prob", "short_prob", "hold_prob"}
    assert (
        abs(result["long_prob"] + result["short_prob"] + result["hold_prob"] - 1.0)
        < 1e-5
    )
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_kronos_save_load(tmp_path):
    from training.kronos_model import KronosClassifier

    model = KronosClassifier(hidden_dim=32, n_heads=2, n_layers=1)
    path = str(tmp_path / "test.pt")
    model.save(path)
    loaded = KronosClassifier.load(path)
    # Outputs must match in eval mode
    x = _make_x(1, 63)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)


def test_kronos_predict_from_kronos_agent_format():
    from training.kronos_model import KronosClassifier

    # Replicate KronosAgent._prepare_input() output: 63 bars × 5 features
    model = KronosClassifier(hidden_dim=32, n_heads=2, n_layers=1)
    klines = [
        {
            "open": 22000 + i,
            "high": 22020 + i,
            "low": 21980 + i,
            "close": 22010 + i,
            "volume": 1_000_000,
        }
        for i in range(64)
    ]
    # Replicate KronosAgent._prepare_input manually
    x = []
    for i in range(1, 64):
        prev = klines[i - 1]["close"]
        x.append(
            [
                (klines[i]["open"] - prev) / prev,
                (klines[i]["high"] - prev) / prev,
                (klines[i]["low"] - prev) / prev,
                (klines[i]["close"] - prev) / prev,
                klines[i]["volume"] / 1e6,
            ]
        )
    result = model.predict(x)
    assert "long_prob" in result
