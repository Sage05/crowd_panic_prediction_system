import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from dashboard_state import DashboardState

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = DashboardState()


@app.get("/api/hello")
def hello():
    return {"message": "backend is alive"}


async def update_live_state():
    while True:
        state.update()
        await asyncio.sleep(1)


@app.on_event("startup")
async def startup():
    state.update()
    asyncio.create_task(update_live_state())


@app.websocket("/ws/live")
async def live(ws: WebSocket):

    await ws.accept()

    try:
        while True:
            await ws.send_json(state.to_dict())
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass