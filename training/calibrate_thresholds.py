import json
import re
from pathlib import Path
from datetime import datetime, timezone

DEFAULTS = {
    "onchain_funding_threshold": -0.001,
    "onchain_ls_ratio_threshold": 0.9,
    "news_sentiment_threshold": 0.2,
}
OUT_PATH_DEFAULT = "data/calibrated_thresholds.json"
MIN_SAMPLES = 50


def _extract_funding_from_transcript(transcript: str) -> float | None:
    match = re.search(r"Funding:\s*([-\d.]+)", transcript or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _extract_sentiment_from_transcript(transcript: str) -> float | None:
    match = re.search(r"Sentiment score:\s*([-\d.]+)", transcript or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def calibrate(
    log_path: str = "data/signal_log.json",
    out_path: str = OUT_PATH_DEFAULT,
    min_samples: int = MIN_SAMPLES,
) -> dict:
    """
    Read resolved crypto signals from signal_log.json.
    Compute threshold values that maximise TARGET_HIT precision.
    Falls back to defaults if < min_samples resolved signals.
    """
    try:
        with open(log_path) as f:
            signals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        signals = []

    crypto_resolved = [
        s
        for s in signals
        if s.get("market") == "crypto"
        and s.get("outcome") in ("TARGET_HIT", "STOP_HIT")
        and s.get("debate_transcript")
    ]

    result = dict(DEFAULTS)
    result["calibrated"] = False
    result["sample_size"] = len(crypto_resolved)

    if len(crypto_resolved) < min_samples:
        result["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save(result, out_path)
        return result

    funding_pairs = []
    sentiment_pairs = []
    for s in crypto_resolved:
        won = s["outcome"] == "TARGET_HIT"
        transcript = s.get("debate_transcript", "")
        funding = _extract_funding_from_transcript(transcript)
        sentiment = _extract_sentiment_from_transcript(transcript)
        if funding is not None:
            funding_pairs.append((funding, won))
        if sentiment is not None:
            sentiment_pairs.append((sentiment, won))

    if len(funding_pairs) >= 20:
        best_thresh, best_precision = DEFAULTS["onchain_funding_threshold"], 0.0
        candidates = sorted(set(f for f, _ in funding_pairs))
        for thresh in candidates:
            positives = [(f, w) for f, w in funding_pairs if f < thresh]
            if len(positives) < 10:
                continue
            precision = sum(w for _, w in positives) / len(positives)
            if precision > best_precision:
                best_precision = precision
                best_thresh = thresh
        result["onchain_funding_threshold"] = best_thresh

    if len(sentiment_pairs) >= 20:
        best_thresh, best_precision = DEFAULTS["news_sentiment_threshold"], 0.0
        candidates = sorted(set(s for s, _ in sentiment_pairs))
        for thresh in candidates:
            positives = [(s, w) for s, w in sentiment_pairs if s > thresh]
            if len(positives) < 10:
                continue
            precision = sum(w for _, w in positives) / len(positives)
            if precision > best_precision:
                best_precision = precision
                best_thresh = thresh
        result["news_sentiment_threshold"] = best_thresh

    result["calibrated"] = True
    result["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save(result, out_path)
    return result


def _save(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    result = calibrate()
    print(json.dumps(result, indent=2))
