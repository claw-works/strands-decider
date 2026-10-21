"""HTTP server exposing the decision API in two wire formats over one engine.

`POST /v1/decisions` follows OpenAI's Decisions API, so the `openai` SDKs work against
this server by changing their base URL (see openai_schema.py for the mapping).

`POST /v1/systemone` follows the public Jev API documentation. JevBench's typesafe adapter
runs against it unchanged (evaluation/jevbench.md). Compatibility with the Jev API itself
is not verified.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .infer import EngineConfig, SystemOneEngine, load_mlx
from .modeling import StrandsDeciderModel
from .openai_schema import Decision, DecisionCreateRequest, error_body, from_systemone, to_systemone
from .schema import SystemOneRequest, SystemOneResponse

_engine: SystemOneEngine | None = None
_engine_lock = threading.Lock()


def _needs_serialisation(device: str) -> bool:
    """MPS aborts the process with a Metal command-buffer assertion (issue #9) when
    two overlapping threads hit the same device. CUDA and CPU are thread-safe for
    inference, and MLX serialises on its own scheduler."""
    return str(device).startswith("mps")


def get_engine() -> SystemOneEngine:
    if _engine is None:  # pragma: no cover - guarded by create_app
        raise RuntimeError("engine not initialised")
    return _engine


def create_app(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    attn_implementation: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
) -> FastAPI:
    global _engine

    app = FastAPI(
        title="strands-decider System One",
        version="0.1.0",
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
    )

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    config = EngineConfig(
        device=device, use_prefix_cache=use_prefix_cache, model_name=resolved_name,
        strict_window=strict_window, max_batch=max_batch, max_batch_tokens=max_batch_tokens,
    )
    if vision:
        if device == "mlx":
            raise ValueError("--vision runs on torch devices (cuda, mps, cpu), not mlx")
        from .vision import load_vision_engine

        _engine = load_vision_engine(checkpoint, config, attn_implementation=attn_implementation)
    elif device == "mlx":
        _engine = load_mlx(checkpoint, config)
    else:
        model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
        _engine = SystemOneEngine(model, config)

    # SageMaker and AgentCore Runtime call /ping and /invocations; each is the same
    # handler as the path above it.
    @app.get("/health")
    @app.get("/ping")
    def health() -> dict:
        eng = get_engine()
        return {
            "status": "ok",
            "model": eng.cfg.model_name,
            "checkpoint": checkpoint,
            "base_model": eng.model.config.base_model,
            "num_slots": eng.model.config.num_slots,
            "max_length": eng.model.config.max_length,
            "temperature": eng.model.config.temperature,
            "device": eng.cfg.device,
            "prefix_cache": eng.cfg.use_prefix_cache,
            "vision": vision,
        }

    guard: contextlib.AbstractContextManager
    guard = _engine_lock if _needs_serialisation(device) else contextlib.nullcontext()

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    @app.post("/invocations", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> JSONResponse:
        eng = get_engine()
        started = time.perf_counter()
        try:
            with guard:
                response = eng.evaluate(request)
        except ValueError as exc:
            # Option count over num_slots, malformed permutation, etc. -- caller error.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = response.model_dump()
        payload["latency_ms"] = round(elapsed_ms, 2)
        return JSONResponse(content=payload)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        # OpenAI clients expect a 400 in OpenAI's error envelope; the System One paths
        # keep FastAPI's 422.
        if request.url.path != "/v1/decisions":
            return await request_validation_exception_handler(request, exc)
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(p) for p in first.get("loc", ()) if p != "body"]
        message = f"{'.'.join(loc) + ': ' if loc else ''}{first.get('msg', 'invalid request')}"
        return JSONResponse(status_code=400, content=error_body(message, param=".".join(loc) or None))

    @app.post("/v1/decisions", response_model=Decision)
    def decisions(request: DecisionCreateRequest) -> JSONResponse:
        eng = get_engine()
        try:
            native = to_systemone(request)
            with guard:
                response = eng.evaluate(native)
        except ValueError as exc:
            # Score levels out of range, more options than slots, images on a text-only
            # engine -- caller error, reported the way OpenAI reports it.
            return JSONResponse(status_code=400, content=error_body(str(exc)))
        return JSONResponse(content=from_systemone(request, response).model_dump())

    return app


def serve(
    checkpoint: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
) -> None:
    import uvicorn

    app = create_app(
        checkpoint,
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=model_name,
        strict_window=strict_window,
        max_batch=max_batch,
        vision=vision,
        max_batch_tokens=max_batch_tokens,
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
