import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.dashboard_state import DashboardState

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = DashboardState()


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Crowd Panic Prediction Backend",
    }


@app.get("/api/hello")
def hello():
    return {
        "message": "backend is alive",
    }


async def update_live_state():

    while True:

        try:
            state.update()

        except Exception as e:
            import traceback

            print(f"[Backend Error] {e}")
            traceback.print_exc()

        await asyncio.sleep(1)


@app.on_event("startup")
async def startup():

    asyncio.create_task(update_live_state())


@app.on_event("shutdown")
async def shutdown():

    state.camera_manager.shutdown()


@app.websocket("/ws/live")
async def live(ws: WebSocket):

    await ws.accept()

    try:

        while True:

            print(state.to_dict().keys())
            print(len(state.to_dict()["cameras"]))
            await ws.send_json(
                state.to_dict()
            )

            await asyncio.sleep(1)

    except WebSocketDisconnect:

        print("Dashboard disconnected.")