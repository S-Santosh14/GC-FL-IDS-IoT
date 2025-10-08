# backend/api/main.py
import json, time, asyncio, random
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "metrics.json"

app = FastAPI(title="GC-FL-IDS Backend", version="1.0.0")

# Allow your Vite dev server and Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

def _read_metrics():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    # default empty payload
    return {
        "dataset": "unknown",
        "feature_selection": {"method":"none","selected_features":[],"mi_ge_0p8":[]},
        "unbalanced": {}, "balanced_oversample": {},
        "attack_distribution_pred": {}, "last_updated": int(time.time())
    }

@app.get("/metrics/latest")
def latest_metrics():
    return JSONResponse(_read_metrics())

@app.get("/metrics/attack-distribution")
def attack_distribution():
    data = _read_metrics().get("attack_distribution_pred", {})
    # Return as {type, count} list to match your UI
    return JSONResponse([{"type": str(k), "count": int(v)} for k, v in data.items()])

# ---- Simple SSE stream that emits alerts every 2s ----
async def alert_stream() -> AsyncGenerator[str, None]:
    idx = 0
    attack_names = ["Normal","DoS","Probe","R2L","U2R","Botnet"]
    while True:
        idx += 1
        payload = {
            "id": idx,
            "nodeId": f"edge-{random.randint(1,4)}",
            "attackType": random.choice(attack_names),
            "confidence": round(random.uniform(0.75, 0.99), 3),
            "latencyMs": random.randint(50, 180),
            "ts": int(time.time()*1000)
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(2)

@app.get("/stream/alerts")
async def stream_alerts():
    return StreamingResponse(alert_stream(), media_type="text/event-stream")
