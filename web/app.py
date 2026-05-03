"""
FastAPI web service for kml2ofds: upload KML, configure profile, convert, download.
"""

import io
import json
import queue
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from kml2ofds.constants import AUTO_GENERATED_NODE_NAME, NETWORK_FORK_NAME
from kml2ofds.rfc4122 import (
    network_id_validation_error,
    optional_rfc4122_uuid_validation_error,
)

# In-memory store: token -> {"zip": bytes, "nodes": str, "spans": str}
_download_store: dict[str, dict] = {}
_store_lock = threading.Lock()

MAX_KML_SIZE = 50 * 1024 * 1024  # 50MB
# Progress is only emitted between pipeline stages; consolidation/export can run
# many minutes without a new message. A short queue timeout caused false
# "Conversion timed out" in the UI while the worker was still healthy.
QUEUE_HEARTBEAT_SEC = 45
MAX_CONVERSION_WALL_SEC = 2 * 60 * 60  # hard cap: 2 hours wall-clock

app = FastAPI(title="kml2ofds Online", version="0.1.0")

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

FAVICON_PATH = WEB_DIR / "static" / "favicon.ico"


@app.api_route(
    "/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False
)
async def favicon():
    """Serve favicon at root (GET); HEAD for metadata-only checks (e.g. curl -I)."""
    if FAVICON_PATH.exists():
        return FileResponse(FAVICON_PATH, media_type="image/x-icon")
    raise HTTPException(404, "Not found")


def _run_conversion_thread(
    kml_content: bytes,
    config_dict: dict[str, str],
    progress_queue: queue.Queue,
) -> None:
    """Run conversion in thread, pushing progress to queue."""
    try:
        from kml2ofds.api import run_conversion

        def progress(stage: int, total: int, message: str) -> None:
            progress_queue.put({"stage": stage, "total": total, "message": message})

        result = run_conversion(
            kml_content,
            config_dict,
            progress_callback=progress,
        )

        # Build ZIP (result keys are prefixed filenames from output_paths)
        import zipfile

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in result.items():
                zf.writestr(name, content)

        nodes_key = next(k for k in result if "ofds-nodes" in k and k.endswith(".geojson"))
        spans_key = next(k for k in result if "ofds-spans" in k and k.endswith(".geojson"))
        token = secrets.token_urlsafe(16)
        nodes_geojson = result[nodes_key].decode("utf-8")
        spans_geojson = result[spans_key].decode("utf-8")
        with _store_lock:
            _download_store[token] = {
                "zip": zip_buffer.getvalue(),
                "nodes": nodes_geojson,
                "spans": spans_geojson,
            }

        progress_queue.put({
            "done": True,
            "download_url": f"/download/{token}",
            "geojson_nodes_url": f"/geojson/{token}/nodes",
            "geojson_spans_url": f"/geojson/{token}/spans",
        })
    except Exception as e:
        progress_queue.put({"error": str(e)})


def _sse_format(data: dict) -> str:
    """Format dict as SSE event."""
    return f"data: {json.dumps(data)}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve upload + profile form."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "auto_generated_node_name": AUTO_GENERATED_NODE_NAME,
            "network_fork_name": NETWORK_FORK_NAME,
        },
    )


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """Serve about / how-it-works page."""
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/convert")
async def convert(
    kml_file: UploadFile = File(...),
    network_name: str = Form("Default Network Name"),
    network_id: str = Form(""),
    network_status: str = Form("Operational"),
    output_name_prefix: str = Form("OFDS"),
    physicalInfrastructureProvider_name: str = Form(""),
    physicalInfrastructureProvider_id: str = Form(""),
    networkProviders_name: str = Form(""),
    networkProviders_id: str = Form(""),
    ignore_placemarks: str = Form(""),
    threshold_meters: str = Form("5000"),
    rename_spans_from_nodes: str = Form("true"),
    merge_contiguous_spans: str = Form("true"),
    merge_contiguous_spans_precision: str = Form("3"),
    merge_proximate_nodes: str = Form(""),
    merge_proximate_nodes_meters: str = Form("50"),
):
    """Accept KML + form; stream progress via SSE; return download URL when done."""
    # Validate file
    if not kml_file.filename or not kml_file.filename.lower().endswith(".kml"):
        raise HTTPException(400, "Please upload a .kml file")

    kml_content = await kml_file.read()
    if len(kml_content) > MAX_KML_SIZE:
        raise HTTPException(
            400,
            f"File too large. Maximum size is {MAX_KML_SIZE // (1024*1024)}MB.",
        )

    # Basic XML check
    try:
        kml_content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be valid UTF-8 text")

    nid_err = network_id_validation_error(network_id)
    if nid_err:
        raise HTTPException(422, nid_err)
    pip_err = optional_rfc4122_uuid_validation_error(
        physicalInfrastructureProvider_id,
        label="Physical infrastructure provider ID",
    )
    if pip_err:
        raise HTTPException(422, pip_err)
    np_err = optional_rfc4122_uuid_validation_error(
        networkProviders_id,
        label="Network providers ID",
    )
    if np_err:
        raise HTTPException(422, np_err)

    config_dict = {
        "kml_file_name": kml_file.filename,
        "network_name": network_name,
        "network_id": network_id or None,
        "network_status": network_status,
        "output_name_prefix": output_name_prefix or "OFDS",
        "physicalInfrastructureProvider_name": physicalInfrastructureProvider_name,
        "physicalInfrastructureProvider_id": physicalInfrastructureProvider_id,
        "networkProviders_name": networkProviders_name,
        "networkProviders_id": networkProviders_id,
        "ignore_placemarks": ignore_placemarks,
        "threshold_meters": threshold_meters,
        "rename_spans_from_nodes": rename_spans_from_nodes,
        "merge_contiguous_spans": merge_contiguous_spans,
        "merge_contiguous_spans_precision": merge_contiguous_spans_precision,
        "merge_proximate_nodes": merge_proximate_nodes,
        "merge_proximate_nodes_meters": merge_proximate_nodes_meters,
    }
    config_dict = {k: (v or "") for k, v in config_dict.items()}

    progress_queue = queue.Queue()

    def event_generator():
        thread = threading.Thread(
            target=_run_conversion_thread,
            args=(kml_content, config_dict, progress_queue),
        )
        thread.start()
        wall_start = time.monotonic()

        while True:
            if time.monotonic() - wall_start > MAX_CONVERSION_WALL_SEC:
                yield _sse_format(
                    {
                        "error": (
                            "Conversion exceeded maximum time limit "
                            f"({MAX_CONVERSION_WALL_SEC // 3600} hours). "
                            "Try simplifying the KML or running the CLI locally."
                        )
                    }
                )
                break

            try:
                event = progress_queue.get(timeout=QUEUE_HEARTBEAT_SEC)
            except queue.Empty:
                if thread.is_alive():
                    # SSE comment: keeps some proxies from closing idle streams.
                    yield ": ping\n\n"
                    continue
                try:
                    event = progress_queue.get_nowait()
                except queue.Empty:
                    yield _sse_format(
                        {
                            "error": (
                                "Conversion ended without result "
                                "(worker stopped unexpectedly)."
                            )
                        }
                    )
                    break

            if "error" in event:
                yield _sse_format(event)
                break
            if event.get("done"):
                yield _sse_format(event)
                break
            yield _sse_format(event)

        thread.join(timeout=5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/download/{token}")
async def download(token: str):
    """Serve ZIP for one-time download; remove from store after."""
    with _store_lock:
        entry = _download_store.pop(token, None)
    if entry is None:
        raise HTTPException(404, "Download not found or expired")
    return Response(
        content=entry["zip"],
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="kml2ofds-output.zip"',
        },
    )


@app.get("/geojson/{token}/nodes")
async def geojson_nodes(token: str):
    """Serve nodes GeoJSON for map preview. Data removed when ZIP is downloaded."""
    with _store_lock:
        entry = _download_store.get(token)
    if entry is None:
        raise HTTPException(404, "Not found or expired")
    return Response(
        content=entry["nodes"],
        media_type="application/geo+json",
    )


@app.get("/geojson/{token}/spans")
async def geojson_spans(token: str):
    """Serve spans GeoJSON for map preview. Data removed when ZIP is downloaded."""
    with _store_lock:
        entry = _download_store.get(token)
    if entry is None:
        raise HTTPException(404, "Not found or expired")
    return Response(
        content=entry["spans"],
        media_type="application/geo+json",
    )


@app.get("/health")
async def health():
    """Health check for deployment."""
    return {"status": "ok"}
