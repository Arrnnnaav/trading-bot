import torch


def test_model_output_shape():
    from training.model import ChronosClassifier

    model = ChronosClassifier(checkpoint="amazon/chronos-t5-small")
    batch = torch.randn(2, 96)  # 2 samples, 96 time steps
    logits = model(batch)
    assert logits.shape == torch.Size([2, 3])


def test_model_freeze_unfreeze():
    from training.model import ChronosClassifier

    model = ChronosClassifier(checkpoint="amazon/chronos-t5-small")
    model.freeze_encoder()
    frozen = sum(1 for p in model.encoder.parameters() if not p.requires_grad)
    assert frozen > 0
    model.unfreeze_encoder()
    unfrozen = sum(1 for p in model.encoder.parameters() if p.requires_grad)
    assert unfrozen > 0


def test_model_save_load(tmp_path):
    from training.model import ChronosClassifier

    model = ChronosClassifier(checkpoint="amazon/chronos-t5-small")
    save_path = str(tmp_path / "test_model.pt")
    model.save(save_path)
    loaded = ChronosClassifier.load(save_path, checkpoint="amazon/chronos-t5-small")
    # Check weights match — eval() disables Dropout so outputs are deterministic
    batch = torch.randn(1, 96)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        out1 = model(batch)
        out2 = loaded(batch)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_predict_returns_valid_direction():
    from training.model import ChronosClassifier
    from core.models import Direction

    model = ChronosClassifier(checkpoint="amazon/chronos-t5-small")
    close = torch.randn(96)
    direction, confidence = model.predict(close)
    assert direction in (Direction.LONG, Direction.SHORT, Direction.HOLD)
    assert 0.0 <= confidence <= 1.0
