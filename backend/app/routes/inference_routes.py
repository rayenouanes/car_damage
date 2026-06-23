from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from backend.app.models.schemas import InferenceRequest


router = APIRouter(prefix="/api", tags=["inference"])


@router.post("/jobs/{job_id}/infer")
def infer_job(job_id: str, payload: InferenceRequest, request: Request) -> dict:
    if payload.mode.value != "training":
        raise HTTPException(400, "Utiliser /api/audit/infer ou /api/production/infer pour ce mode")
    try:
        return request.app.state.training_pipeline.run(job_id, payload.force)
    except KeyError:
        raise HTTPException(404, "Job introuvable") from None
    except Exception as exc:
        raise HTTPException(500, f"Echec inference: {exc}") from exc


@router.get("/images/{image_id}/content")
def image_content(image_id: str, request: Request) -> FileResponse:
    image = request.app.state.database.get_image(image_id)
    if not image:
        raise HTTPException(404, "Image introuvable")
    path = Path(image["path"])
    if not path.exists():
        raise HTTPException(404, "Fichier image absent")
    return FileResponse(path)


@router.get("/review-queue")
def review_queue(request: Request, job_id: str | None = None, limit: int = 200) -> list[dict]:
    return request.app.state.database.review_queue(job_id, min(max(limit, 1), 1000))


@router.post("/production/infer")
async def production_inference(request: Request, file: UploadFile = File(...)) -> list[dict]:
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=request.app.state.settings.path("uploads"), suffix=suffix, delete=False
        ) as temporary:
            temporary.write(await file.read())
            temp_path = Path(temporary.name)
        import cv2

        pixels = cv2.imread(str(temp_path))
        if pixels is None:
            raise ValueError("Image illisible")
        height, width = pixels.shape[:2]
        image = {"path": str(temp_path), "width": width, "height": height}
        detections = request.app.state.yolo_service.adapter.predict(temp_path)
        clean = [
            {"class_name": item["class_name"], "bbox": item["bbox"], "score": item["confidence"]}
            for item in detections
        ]
        return clean
    except Exception as exc:
        raise HTTPException(500, f"Echec inference production: {exc}") from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


@router.post("/audit/infer")
async def audit_inference(
    request: Request,
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.70),
) -> dict:
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=request.app.state.settings.path("uploads"), suffix=suffix, delete=False
        ) as temporary:
            temporary.write(await file.read())
            temp_path = Path(temporary.name)
        import cv2

        pixels = cv2.imread(str(temp_path))
        if pixels is None:
            raise ValueError("Image illisible")
        height, width = pixels.shape[:2]
        image = {
            "id": "audit", "job_id": "audit", "path": str(temp_path),
            "width": width, "height": height,
        }
        detections = request.app.state.yolo_service.predict(image)
        audited = []
        for detection in detections:
            item = {"id": "audit", **detection}
            is_doubtful = (
                detection["confidence"] < confidence_threshold
                or "classes_confondues" in detection["active_learning_reasons"]
                or "proche_zone_brillante" in detection["active_learning_reasons"]
            )
            vlm_payload = None
            llm_payload = None
            if is_doubtful:
                vlm, rules_text = request.app.state.vlm_service.analyze(image, item, None)
                llm = request.app.state.llm_service.critique(item, None, vlm, rules_text)
                vlm_payload = vlm.model_dump(mode="json")
                llm_payload = llm.model_dump(mode="json")
            audited.append({**detection, "audited": is_doubtful, "vlm": vlm_payload, "llm": llm_payload})
        return {"mode": "audit", "detections": audited}
    except Exception as exc:
        raise HTTPException(500, f"Echec audit: {exc}") from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
