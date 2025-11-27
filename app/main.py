from fastapi import FastAPI, WebSocket
from app.graph.workflow import  build_workflow
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
workflow = build_workflow()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        print(f"[WS] Received question: {data}")

        initial_state = {"question": data, "docs": [], "answer": ""}

        final_state = workflow.invoke(initial_state)
        answer = final_state["answer"]
        ans_string = str(answer)
        print(f"[WS] Sending answer: {ans_string}")
        await websocket.send_text(ans_string)
