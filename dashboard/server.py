import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
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


@app.get("/positions")
async def positions_redirect():
    return RedirectResponse(url="/", status_code=301)


@app.get("/trades", response_class=HTMLResponse)
async def trades():
    return Path("dashboard/static/trades.html").read_text()


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


@app.get("/api/trades")
async def get_trades(limit: int = 200):
    signals = aggregator.get_all_signals()
    executed = [s for s in signals if s.get("executed")]
    trades = []
    for s in executed[-limit:]:
        trades.append(
            {
                "id": s["id"],
                "ticker": s["ticker"],
                "direction": s["direction"],
                "entry_price": s["entry_price"],
                "target_price": s["target_price"],
                "stop_price": s["stop_price"],
                "entry_time": s.get("generated_at"),
                "exit_time": s.get("outcome_at"),
                "exit_price": s.get("outcome_price"),
                "outcome": s.get("outcome", "PENDING"),
                "pnl_inr": round(s.get("hypothetical_pnl_inr") or 0.0, 2),
                "confidence": s.get("confidence", 0.0),
            }
        )
    return trades


@app.get("/api/equity-curve")
async def get_equity_curve():
    signals = aggregator.get_all_signals()
    resolved = [
        s
        for s in signals
        if s.get("executed")
        and s.get("outcome_at")
        and s.get("hypothetical_pnl_inr") is not None
        and s.get("outcome") not in ("PENDING", None)
    ]
    resolved.sort(key=lambda s: s.get("outcome_at") or "")
    cumulative = 0.0
    points = []
    for s in resolved:
        cumulative += s["hypothetical_pnl_inr"]
        points.append({"t": s["outcome_at"], "pnl": round(cumulative, 2)})
    return points


@app.get("/api/stats")
async def get_stats():
    signals = aggregator.get_all_signals()
    executed = [s for s in signals if s.get("executed")]
    resolved = [s for s in executed if s.get("outcome") not in ("PENDING", None)]
    wins = [s for s in resolved if s["outcome"] == "TARGET_HIT"]
    losses = [s for s in resolved if s["outcome"] == "STOP_HIT"]
    win_pnls = [
        s["hypothetical_pnl_inr"]
        for s in wins
        if s.get("hypothetical_pnl_inr") is not None
    ]
    loss_pnls = [
        s["hypothetical_pnl_inr"]
        for s in losses
        if s.get("hypothetical_pnl_inr") is not None
    ]
    total_pnl = sum(s.get("hypothetical_pnl_inr") or 0 for s in resolved)

    cumulative, peak, max_dd = 0.0, 0.0, 0.0
    for s in sorted(resolved, key=lambda x: x.get("outcome_at") or ""):
        cumulative += s.get("hypothetical_pnl_inr") or 0
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    return {
        "total_trades": len(executed),
        "resolved": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(resolved) * 100, 1) if resolved else 0.0,
        "total_pnl_inr": round(total_pnl, 2),
        "avg_win_inr": round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0,
        "avg_loss_inr": round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0,
        "max_drawdown_inr": round(max_dd, 2),
    }


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
