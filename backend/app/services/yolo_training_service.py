from __future__ import annotations

from backend.app.config import Settings
from backend.app.models.schemas import YoloTrainingRequest


class YoloTrainingService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def train(self, export_id: str, request: YoloTrainingRequest) -> dict:
        dataset_dir = self.settings.path("exports") / export_id
        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Export introuvable: {export_id}")
        parameters = {
            "data": str(data_yaml.resolve()),
            "epochs": request.epochs,
            "imgsz": request.image_size,
            "batch": request.batch,
            "device": request.device,
            "project": str(self.settings.path("training_runs")),
            "name": export_id,
        }
        if not self.settings.yolo_training_enabled:
            return {
                "status": "prepared",
                "message": "Activer YOLO_TRAINING_ENABLED=true pour lancer l'entrainement.",
                "model": str(self.settings.yolo_model_path),
                "parameters": parameters,
            }
        if not self.settings.yolo_model_path.exists():
            raise FileNotFoundError("Un poids YOLO local est requis pour le reentrainement")
        from ultralytics import YOLO

        model = YOLO(str(self.settings.yolo_model_path))
        result = model.train(task="segment", **parameters)
        return {"status": "completed", "save_dir": str(result.save_dir), "parameters": parameters}

