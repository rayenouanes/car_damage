from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.adapters.vector_store_chroma import ChromaVectorStore
from backend.app.config import CANONICAL_CLASSES, Settings
from backend.app.database import Database
from backend.app.routes import (
    correction_routes,
    export_routes,
    inference_routes,
    rag_routes,
    upload_routes,
)
from backend.app.services.correction_service import CorrectionService
from backend.app.services.export_yolo_service import ExportYoloService
from backend.app.services.frame_selection_service import FrameSelectionService
from backend.app.services.llm_critic_service import LLMCriticService
from backend.app.services.rag_service import RAGService
from backend.app.services.sam2_service import SAM2Service
from backend.app.services.training_pipeline_service import TrainingPipelineService
from backend.app.services.video_service import VideoService
from backend.app.services.vlm_teacher_service import VLMTeacherService
from backend.app.services.yolo_service import YoloService
from backend.app.services.yolo_training_service import YoloTrainingService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    rag_service = RAGService(
        ChromaVectorStore(settings.path("rag"), prefer_chroma=settings.use_chroma)
    )
    rag_service.seed_initial_rules()

    app = FastAPI(
        title="Automotive Defect Active Learning MVP",
        version="0.2.0",
        description="YOLO production inference and a separate human-in-the-loop learning pipeline.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
        allow_methods=["*"], allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.database = database
    app.state.video_service = VideoService(database, settings)
    app.state.yolo_service = YoloService(database, settings)
    app.state.rag_service = rag_service
    app.state.vlm_service = VLMTeacherService(database, rag_service)
    app.state.llm_service = LLMCriticService(database)
    app.state.sam2_service = SAM2Service(settings)
    app.state.frame_selection_service = FrameSelectionService(database, settings)
    app.state.training_pipeline = TrainingPipelineService(
        database, app.state.frame_selection_service, app.state.yolo_service,
        app.state.sam2_service, app.state.vlm_service, app.state.llm_service,
    )
    app.state.correction_service = CorrectionService(database, settings, rag_service)
    app.state.export_service = ExportYoloService(database, settings)
    app.state.yolo_training_service = YoloTrainingService(settings)

    for router in (
        upload_routes.router,
        inference_routes.router,
        correction_routes.router,
        rag_routes.router,
        export_routes.router,
    ):
        app.include_router(router)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "yolo_provider": app.state.yolo_service.provider,
            "model_path": str(settings.yolo_model_path),
            "model_exists": settings.yolo_model_path.exists(),
            "vlm_provider": app.state.vlm_service.provider,
            "llm_provider": app.state.llm_service.provider,
            "sam2_enabled": settings.sam2_enabled,
            "sam2_provider": app.state.sam2_service.provider,
            "classes": list(CANONICAL_CLASSES),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
