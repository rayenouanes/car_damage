from __future__ import annotations

import json
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_temp_best_2" / "weights_20260625_095612"
CHECKPOINT = RUN_DIR / "weights" / "last.pt"
STATUS_PATH = ROOT / "training_status.json"
ERROR_PATH = ROOT / "last_training_error.txt"
LOG_MARKER = ROOT / "resume_training_21_to_40.marker"


def read_epoch_count() -> int:
    results_csv = RUN_DIR / "results.csv"
    if not results_csv.exists():
        return 0
    lines = [line for line in results_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(0, len(lines) - 1)


def save_status(payload: dict) -> None:
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATUS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def gpu_info() -> dict:
    info = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available() and torch.cuda.device_count() > 0),
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["memory_total_gb"] = round(props.total_memory / 1024**3, 2)
        info["memory_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1024**3, 2)
    return info


def export_finished_weights() -> None:
    exports = {
        RUN_DIR / "weights" / "best.pt": ROOT / "data7_from_best_2_40epochs_resumed_best.pt",
        RUN_DIR / "weights" / "last.pt": ROOT / "data7_from_best_2_40epochs_resumed_last.pt",
        RUN_DIR / "results.csv": ROOT / "results_data7_from_best_2_40epochs_resumed_best.csv",
    }
    for source, target in exports.items():
        if source.exists():
            shutil.copy2(source, target)
    csv_source = RUN_DIR / "results.csv"
    csv_last = ROOT / "results_data7_from_best_2_40epochs_resumed_last.csv"
    if csv_source.exists():
        shutil.copy2(csv_source, csv_last)


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {CHECKPOINT}")

    start_epoch = read_epoch_count()
    save_status(
        {
            "state": "running",
            "model": "best_2.pt",
            "resume_from": str(CHECKPOINT),
            "current_epoch": start_epoch,
            "total_epochs": 40,
            "progress": start_epoch / 40 if start_epoch else 0.0,
            "metrics": {},
            "losses": {},
            "gpu": gpu_info(),
            "note": "Reprise de l'entrainement depuis le checkpoint last.pt de l'epoch 21.",
        }
    )
    LOG_MARKER.write_text(
        f"resume_started={datetime.now().isoformat(timespec='seconds')}\n"
        f"checkpoint={CHECKPOINT}\n"
        f"start_epoch={start_epoch}\n",
        encoding="utf-8",
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(str(CHECKPOINT))

    def update_progress(trainer) -> None:
        epoch = read_epoch_count()
        metrics = {}
        losses = {}
        try:
            metrics = {
                str(k): float(v)
                for k, v in getattr(trainer, "metrics", {}).items()
                if isinstance(v, (int, float))
            }
        except Exception:
            metrics = {}
        try:
            label_loss_items = getattr(trainer, "label_loss_items", None)
            if callable(label_loss_items):
                losses = {str(k): float(v) for k, v in label_loss_items().items()}
        except Exception:
            losses = {}
        save_status(
            {
                "state": "running",
                "model": "best_2.pt",
                "resume_from": str(CHECKPOINT),
                "current_epoch": epoch,
                "total_epochs": 40,
                "progress": min(1.0, epoch / 40),
                "metrics": metrics,
                "losses": losses,
                "gpu": gpu_info(),
                "note": "Reprise de l'entrainement depuis le checkpoint last.pt de l'epoch 21.",
            }
        )

    model.add_callback("on_train_epoch_end", update_progress)

    try:
        model.train(resume=True)
        final_epoch = read_epoch_count()
        export_finished_weights()
        save_status(
            {
                "state": "completed",
                "model": "best_2.pt",
                "resume_from": str(CHECKPOINT),
                "current_epoch": final_epoch,
                "total_epochs": 40,
                "progress": min(1.0, final_epoch / 40),
                "metrics": {},
                "losses": {},
                "gpu": gpu_info(),
                "note": "Reprise terminee. Les poids resumed best/last ont ete exportes a la racine du projet.",
            }
        )
    except Exception as exc:
        ERROR_PATH.write_text(traceback.format_exc(), encoding="utf-8")
        failed_epoch = read_epoch_count()
        save_status(
            {
                "state": "failed",
                "model": "best_2.pt",
                "resume_from": str(CHECKPOINT),
                "current_epoch": failed_epoch,
                "total_epochs": 40,
                "progress": min(1.0, failed_epoch / 40),
                "metrics": {},
                "losses": {},
                "gpu": gpu_info(),
                "error": str(exc),
            }
        )
        raise


if __name__ == "__main__":
    main()
