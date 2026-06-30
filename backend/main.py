import asyncio, json, random
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/hello")
def hello():
    return {"message": "backend is alive"}

def fake_payload():
    """Stand-in for real ML output — same shape the dashboard expects."""
    panic = random.randint(45, 70)
    cams = []
    for i in range(6):
        load = random.randint(20, 80)
        cams.append({
            "id": f"Camera_{i+1:03d}",
            "zone": ["North Stand", "Gate A", "Concourse", "South Ramp", "East Wing", "Plaza"][i],
            "status": "LIVE",
            "riskScore": load / 100,
            "load": load,
            "count": random.randint(80, 400),
            "density": round(random.uniform(0.8, 3.5), 1),
            "isAnomaly": load > 72,
            "grid": [[random.random() for _ in range(12)] for _ in range(7)],
        })
    return {
        "site": "Live deployment",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fps": 30,
        "confidence": 94,
        "systemStatus": "Elevated",
        "riskClass": "Moderate",
        "panicRisk": panic,
        "peopleCount": sum(c["count"] for c in cams),
        "capacity": 8000,
        "trend": 16,
        "cameras": cams,
        "zones": [{"name": c["zone"], "load": c["load"]} for c in cams],
        "alerts": [{"time": "14:33:02", "level": "OK", "zone": "System", "msg": "Backend connected"}],
        "panicHistory": [random.uniform(0.3, 0.6) for _ in range(30)],
        "densityGrid": [[random.random() for _ in range(22)] for _ in range(6)],
    }

@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps(fake_payload()))
            await asyncio.sleep(1)   # push new data every second
    except WebSocketDisconnect:
        pass