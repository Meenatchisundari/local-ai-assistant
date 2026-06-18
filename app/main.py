import json
from typing import Any
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

OLLAMA_BASE = "http://127.0.0.1:11434"
HTTPX_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0)

app = FastAPI(title="Local AI Assistant", version="0.1.0")

class ChatRequest(BaseModel):
    model: str = Field(..., examples=["tinyllama"])
    prompt: str = Field(..., min_length=1)
    stream: bool = True
    options: dict[str, Any] = Field(default_factory=dict)

@app.get("/health")
def health():
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3.0)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {"status": "ok", "ollama_reachable": ollama_ok}

@app.get("/models")
def list_models():
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return {"models": models}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}")

@app.post("/chat")
async def chat(req: ChatRequest):
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "stream": req.stream,
        "options": req.options,
    }
    if req.stream:
        async def _stream():
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                async with client.stream("POST", f"{OLLAMA_BASE}/api/generate", json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"data: {json.dumps({'error': body.decode()})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
