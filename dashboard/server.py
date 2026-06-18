import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from core.signal_aggregator import SignalAggregator
from core.json_store import read_json
from config import config

app = FastAPI(title="Trading Bot Dashboard")
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

aggregator = SignalAggregator(config.signal_log_path)
_ws_clients: list[WebSocket] = []


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("dashboard/static/index.html").read_text()


@app.get("/history", response_class=HTMLResponse)
async def history():
    return Path("dashboard/static/history.html").read_text()


@app.get("/performance", response_class=HTMLResponse)
async def performance():
    return Path("dashboard/static/performance.html").read_text()


@app.get("/positions", response_class=HTMLResponse)
async def positions():
    return Path("dashboard/static/positions.html").read_text()


@app.get("/api/signals")
async def get_signals(market: str = None, outcome: str = None, limit: int = 200):
    signals = aggregator.get_all_signals()
    if market:
        signals = [s for s in signals if s["market"] == market]
    if outcome:
        signals = [s for s in signals if s["outcome"] == outcome]
    return signals[-limit:]


@app.get("/api/performance")
async def get_performance():
    signals = aggregator.get_all_signals()
    resolved = [s for s in signals if s["outcome"] != "PENDING"]
    wins = [s for s in resolved if s["outcome"] == "TARGET_HIT"]
    losses = [s for s in resolved if s["outcome"] == "STOP_HIT"]
    total_hyp_pnl = sum(s.get("hypothetical_pnl_inr") or 0 for s in resolved)
    executed = [s for s in signals if s["executed"]]
    actual_pnl = sum(
        s.get("hypothetical_pnl_inr") or 0
        for s in executed
        if s["outcome"] != "PENDING"
    )
    return {
        "total_signals": len(signals),
        "resolved": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(resolved) if resolved else 0,
        "total_hypothetical_pnl_inr": round(total_hyp_pnl, 2),
        "total_actual_pnl_inr": round(actual_pnl, 2),
        "signals_by_market": {
            "india": len([s for s in signals if s["market"] == "india"]),
        },
    }


@app.get("/api/open-positions")
async def get_open_positions():
    positions = []
    for path in [config.india_progress_path]:
        try:
            state = read_json(path, {})
            positions.extend(state.get("open_positions", []))
        except Exception:
            pass
    return positions


@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


async def broadcast_signal(signal_dict: dict):
    for ws in list(_ws_clients):
        try:
            await ws.send_json({"type": "signal", "data": signal_dict})
        except Exception:
            if ws in _ws_clients:
                _ws_clients.remove(ws)
