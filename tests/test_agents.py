

def make_klines(n=100):
    return [
        {
            "open": 22000.0,
            "high": 22500.0,
            "low": 21800.0,
            "close": 22400.0 + i * 5,
            "volume": 500000.0,
        }
        for i in range(n)
    ]
