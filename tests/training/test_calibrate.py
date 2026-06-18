import json
from pathlib import Path


def make_signal_log(tmp_path, entries):
    path = tmp_path / "signal_log.json"
    path.write_text(json.dumps(entries))
    return str(path)


def test_calibrate_returns_defaults_if_too_few_signals(tmp_path):
    from training.calibrate_thresholds import calibrate

    log = make_signal_log(tmp_path, [])
    result = calibrate(log_path=log, min_samples=50)
    assert result["onchain_funding_threshold"] == -0.001
    assert result["news_sentiment_threshold"] == 0.2
    assert result["calibrated"] is False


def test_calibrate_writes_json(tmp_path):
    from training.calibrate_thresholds import calibrate

    signals = []
    for i in range(60):
        signals.append(
            {
                "id": f"s{i}",
                "market": "india",
                "direction": "LONG" if i % 2 == 0 else "SHORT",
                "outcome": "TARGET_HIT" if i % 3 == 0 else "STOP_HIT",
                "debate_transcript": f"OnChain: LONG (conf=0.75) — Funding: {-0.0012:.4f} | L/S ratio: 0.80",
            }
        )
    log = make_signal_log(tmp_path, signals)
    out_path = str(tmp_path / "calibrated_thresholds.json")
    result = calibrate(log_path=log, out_path=out_path, min_samples=50)
    assert Path(out_path).exists()
    data = json.loads(Path(out_path).read_text())
    assert "onchain_funding_threshold" in data
    assert "news_sentiment_threshold" in data
