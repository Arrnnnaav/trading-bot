import pandas as pd


def make_df(closes):
    return pd.DataFrame(
        {
            "open_time": range(len(closes)),
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    )


def test_label_long():
    from training.label_data import label_candles

    # +1.5% rise in next 4 candles, no -0.5% stop hit first
    closes = [100.0] + [101.5] * 4 + [100.0] * 10
    df = make_df(closes)
    result = label_candles(df, target_pct=0.01, stop_pct=0.005, window=4)
    assert result["label"].iloc[0] == "LONG"


def test_label_short():
    from training.label_data import label_candles

    # -1.5% drop in next 4 candles, no +0.5% stop hit first
    closes = [100.0] + [98.5] * 4 + [100.0] * 10
    df = make_df(closes)
    result = label_candles(df, target_pct=0.01, stop_pct=0.005, window=4)
    assert result["label"].iloc[0] == "SHORT"


def test_label_hold():
    from training.label_data import label_candles

    # Flat price — neither threshold reached
    closes = [100.0] * 20
    df = make_df(closes)
    result = label_candles(df, target_pct=0.01, stop_pct=0.005, window=4)
    assert result["label"].iloc[0] == "HOLD"


def test_stop_hit_before_target_is_hold():
    from training.label_data import label_candles

    # Drops -0.5% before rising +1% — stop hit first, label = HOLD
    closes = [100.0, 99.4, 99.4, 99.4, 101.5] + [100.0] * 10
    df = make_df(closes)
    result = label_candles(df, target_pct=0.01, stop_pct=0.005, window=4)
    assert result["label"].iloc[0] == "HOLD"


def test_last_rows_are_hold():
    from training.label_data import label_candles

    closes = [100.0] * 20
    df = make_df(closes)
    result = label_candles(df)
    # Last 4 rows can't have full look-ahead window — labeled HOLD
    assert (result["label"].iloc[-4:] == "HOLD").all()
