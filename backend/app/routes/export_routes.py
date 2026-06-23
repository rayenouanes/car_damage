from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from backend.app.models.schemas import ExportSplitRequest, YoloTrainingRequest


router = APIRouter(prefix="/api", tags=["export-yolo"])


@router.post("/jobs/{job_id}/export-yolo")
def export_yolo(job_id: str, split: ExportSplitRequest, request: Request) -> dict:
    try:
        return request.app.state.export_service.export(job_id, split)
    except KeyError:
        raise HTTPException(404, "Job introuvable") from None


@router.get("/exports/{archive_name}")
def download_export(archive_name: str, request: Request) -> FileResponse:
    safe_name = Path(archive_name).name
    if safe_name != archive_name or not safe_name.endswith(".zip"):
        raise HTTPException(400, "Archive invalide")
    path = request.app.state.settings.path("exports") / safe_name
    if not path.exists():
        raise HTTPException(404, "Archive introuvable")
    return FileResponse(path, media_type="application/zip", filename=safe_name)


@router.post("/exports/{export_id}/train-yolo")
def train_yolo(export_id: str, payload: YoloTrainingRequest, request: Request) -> dict:
    try:
        return request.app.state.yolo_training_service.train(export_id, payload)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
