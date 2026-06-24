import os
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from collections import defaultdict

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from dataset_split import prepare_train_val_test_split, summarize_dataset_split, sync_existing_split_item

BEST2_CLASS_CATALOG = ['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']
DEFAULT_CLASS_CATALOG = BEST2_CLASS_CATALOG
CLASS_TRANSLATION_BASE = {
    'dent': 'Bosse (Dent)',
    'scratch': 'Rayure (Scratch)',
    'crack': 'Fissure (Crack)',
    'glass shatter': 'Vitre brisée (Glass shatter)',
    'lamp broken': 'Feu/Phare cassé (Lamp broken)',
    'tire flat': 'Pneu crevé (Tire flat)',
    'reflection': 'Reflet (Reflection)',
    'dust': 'Poussière/Saleté (Dust)'
}
DEFAULT_COLOR_PALETTE = ['#F97066', '#5B9CF6', '#63D0A8', '#FBBF24', '#C084FC', '#FB923C', '#94A3B8', '#A16207']
FINAL_DATASET_CLASSES = {
    0: 'crack',
    1: 'dent',
    2: 'glass shatter',
    3: 'lamp broken',
    4: 'scratch',
    5: 'tire flat'
}
FINAL_DATASET_NAME_TO_ID = {name: cid for cid, name in FINAL_DATASET_CLASSES.items()}
FINAL_DATASET_COLORS = {
    0: '#FFFFFF',
    1: '#F97066',
    2: '#FFFFFF',
    3: '#FFFFFF',
    4: '#C084FC',
    5: '#FFFFFF',
}

APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(APP_BASE_DIR, ".env"))
except ImportError:
    pass

BACKOFFICE_USERNAME = os.getenv("BACKOFFICE_USERNAME", "admin")
BACKOFFICE_PASSWORD = os.getenv("BACKOFFICE_PASSWORD", "")
TRAINING_STATUS_PATH = os.path.join(APP_BASE_DIR, "training_status.json")
SAM2_PACKAGE_ROOT = os.getenv(
    "SAM2_PACKAGE_ROOT",
    r"C:\Users\p134929\Downloads\les modèles (Filip)\model (2)",
)
SAM2_CHECKPOINT_PATH = os.getenv(
    "SAM2_CHECKPOINT",
    os.path.join(SAM2_PACKAGE_ROOT, "sam2.1_hiera_tiny.pt"),
)
SAM2_CONFIG_NAME = os.getenv(
    "SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml"
)


def cuda_is_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def get_gpu_environment_info() -> dict:
    info = {
        "physical_gpu": None,
        "driver": None,
        "torch_version": None,
        "torch_cuda": None,
        "cuda_available": False,
        "device_name": None,
        "memory_total_gb": None,
        "memory_reserved_gb": None,
    }
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if query.returncode == 0 and query.stdout.strip():
            parts = [part.strip() for part in query.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                info["physical_gpu"] = parts[0]
                info["driver"] = parts[1]
                info["memory_total_gb"] = round(float(parts[2]) / 1024, 2)
    except Exception:
        pass
    try:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["torch_cuda"] = str(torch.version.cuda) if torch.version.cuda else None
        info["cuda_available"] = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            info["memory_total_gb"] = round(properties.total_memory / 1024 ** 3, 2)
            info["memory_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 2)
    except Exception:
        pass
    return info


def save_training_status(payload: dict) -> None:
    data = dict(payload)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    temp_path = TRAINING_STATUS_PATH + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as status_file:
            json.dump(data, status_file, ensure_ascii=False, indent=2)
        os.replace(temp_path, TRAINING_STATUS_PATH)
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def load_training_status() -> dict:
    if os.path.exists(TRAINING_STATUS_PATH):
        try:
            with open(TRAINING_STATUS_PATH, "r", encoding="utf-8") as status_file:
                return json.load(status_file)
        except Exception:
            pass
    candidates = []
    for root_name in ("runs_temp_best_2", "runs"):
        root_path = os.path.join(APP_BASE_DIR, root_name)
        if not os.path.isdir(root_path):
            continue
        for current_root, _, filenames in os.walk(root_path):
            if "results.csv" in filenames:
                csv_path = os.path.join(current_root, "results.csv")
                candidates.append((os.path.getmtime(csv_path), csv_path))
    if not candidates:
        return {}
    _, latest_csv = max(candidates, key=lambda item: item[0])
    try:
        with open(latest_csv, "r", encoding="utf-8", newline="") as results_file:
            rows = list(csv.DictReader(results_file))
        if not rows:
            return {}
        last_row = rows[-1]
        current_epoch = int(float(last_row.get("epoch", len(rows)) or len(rows)))
        total_epochs = current_epoch
        args_path = os.path.join(os.path.dirname(latest_csv), "args.yaml")
        if os.path.exists(args_path):
            with open(args_path, "r", encoding="utf-8") as args_file:
                for line in args_file:
                    if line.strip().startswith("epochs:"):
                        total_epochs = int(line.split(":", 1)[1].strip())
                        break
        metrics = {}
        losses = {}
        for key, value in last_row.items():
            clean_key = str(key).strip()
            try:
                numeric_value = round(float(value), 6)
            except Exception:
                continue
            if clean_key.startswith("metrics/") or clean_key.startswith("val/"):
                metrics[clean_key] = numeric_value
            elif clean_key.startswith("train/"):
                losses[clean_key] = numeric_value
        return {
            "state": "completed" if current_epoch >= total_epochs else "interrupted",
            "model": "best_2.pt",
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "progress": min(1.0, current_epoch / total_epochs) if total_epochs else 0.0,
            "metrics": metrics,
            "losses": losses,
            "gpu": get_gpu_environment_info(),
            "updated_at": datetime.fromtimestamp(os.path.getmtime(latest_csv)).isoformat(
                timespec="seconds"
            ),
            "run_path": os.path.relpath(os.path.dirname(latest_csv), APP_BASE_DIR),
        }
    except Exception:
        return {}


def trainer_status_payload(trainer, state: str, model_name: str) -> dict:
    current_epoch = int(getattr(trainer, "epoch", -1)) + 1
    total_epochs = int(getattr(trainer, "epochs", 0) or 0)
    metrics = {}
    for key, value in (getattr(trainer, "metrics", {}) or {}).items():
        try:
            metrics[str(key)] = round(float(value), 6)
        except Exception:
            continue
    losses = {}
    loss_values = getattr(trainer, "tloss", None)
    loss_names = list(getattr(trainer, "loss_names", []) or [])
    try:
        raw_losses = loss_values.detach().cpu().tolist() if hasattr(loss_values, "detach") else list(loss_values)
        if not isinstance(raw_losses, list):
            raw_losses = [raw_losses]
        losses = {
            str(loss_names[index] if index < len(loss_names) else f"loss_{index}"): round(float(value), 6)
            for index, value in enumerate(raw_losses)
        }
    except Exception:
        losses = {}
    gpu_info = get_gpu_environment_info()
    return {
        "state": state,
        "model": model_name,
        "current_epoch": max(0, current_epoch),
        "total_epochs": total_epochs,
        "progress": min(1.0, current_epoch / total_epochs) if total_epochs else 0.0,
        "metrics": metrics,
        "losses": losses,
        "gpu": gpu_info,
    }


def patch_ultralytics_cache_pool_for_windows() -> None:
    try:
        import ultralytics.data.dataset as yolo_dataset

        class SequentialPool:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func=None, iterable=None, chunksize=1):
                if func is None or iterable is None:
                    return iter(())
                return map(func, iterable)

        yolo_dataset.ThreadPool = SequentialPool
        yolo_dataset.NUM_THREADS = 1
    except Exception:
        pass


def humanize_class_name(raw_name: str) -> str:
    if not raw_name:
        return "Classe inconnue"
    normalized = CLASS_TRANSLATION_BASE.get(raw_name)
    if normalized:
        return normalized
    cleaned = raw_name.replace('_', ' ').strip()
    return cleaned.title() if cleaned else raw_name


def normalize_damage_family(raw_name: str):
    """Map model-specific labels to the stable dataset classes."""
    if not raw_name:
        return None
    cleaned = raw_name.lower().replace("_", " ").replace("-", " ").strip()
    compact = " ".join(cleaned.split())
    if compact in FINAL_DATASET_NAME_TO_ID:
        return compact
    if "glass" in compact or "windscreen" in compact or "windshield" in compact:
        return "glass shatter"
    if "lamp" in compact or "headlight" in compact or "taillight" in compact or "signlight" in compact:
        return "lamp broken"
    if "tire" in compact or "tyre" in compact:
        return "tire flat"
    if "scratch" in cleaned or "rayure" in cleaned:
        return "scratch"
    if "dent" in cleaned or "bosse" in cleaned:
        return "dent"
    if "crack" in cleaned or "fissure" in cleaned:
        return "crack"
    return None


def canonical_class_id_from_name(raw_name: str):
    family = normalize_damage_family(raw_name)
    if family is None:
        return None
    return FINAL_DATASET_NAME_TO_ID.get(family)


def get_dataset_class_name(class_id: int) -> str:
    return FINAL_DATASET_CLASSES.get(int(class_id), f"class_{class_id}")


def get_dataset_class_display(class_id: int) -> str:
    class_id = int(class_id)
    if class_id not in FINAL_DATASET_CLASSES:
        return f"Classe non standard #{class_id}"
    return humanize_class_name(FINAL_DATASET_CLASSES[class_id])


def get_dataset_class_color(class_id: int) -> str:
    class_id = int(class_id)
    return FINAL_DATASET_COLORS.get(class_id, DEFAULT_COLOR_PALETTE[class_id % len(DEFAULT_COLOR_PALETTE)])


def find_nonstandard_class_ids(boxes: list) -> list:
    invalid_ids = set()
    for item in boxes:
        try:
            cls_id = int(item.get("class_id"))
        except Exception:
            continue
        if cls_id not in FINAL_DATASET_CLASSES:
            invalid_ids.add(cls_id)
    return sorted(invalid_ids)


def order_annotation_class_ids(class_ids: list[int]) -> list[int]:
    priority_names = ("dent", "scratch")
    priority_ids = [
        FINAL_DATASET_NAME_TO_ID[name]
        for name in priority_names
        if name in FINAL_DATASET_NAME_TO_ID
    ]
    ordered = [cid for cid in priority_ids if cid in class_ids]
    ordered.extend(cid for cid in class_ids if cid not in ordered)
    return ordered


def canvas_rect_to_image_box(rect: dict, scale_factor: float, image_w: int, image_h: int):
    left = float(rect.get("left", 0.0))
    top = float(rect.get("top", 0.0))
    width = float(rect.get("width", 0.0)) * float(rect.get("scaleX", 1.0))
    height = float(rect.get("height", 0.0)) * float(rect.get("scaleY", 1.0))
    if abs(width) <= 10 or abs(height) <= 10 or scale_factor <= 0:
        return None

    canvas_x1, canvas_x2 = sorted((left, left + width))
    canvas_y1, canvas_y2 = sorted((top, top + height))
    x1 = int(round(canvas_x1 / scale_factor))
    y1 = int(round(canvas_y1 / scale_factor))
    x2 = int(round(canvas_x2 / scale_factor))
    y2 = int(round(canvas_y2 / scale_factor))

    x1, x2 = max(0, min(x1, image_w)), max(0, min(x2, image_w))
    y1, y2 = max(0, min(y1, image_h)), max(0, min(y2, image_h))
    if (x2 - x1) < 3 or (y2 - y1) < 3:
        return None
    return [x1, y1, x2, y2]


def normalize_model_names(raw_names) -> dict:
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    return {}


def sync_active_model_classes(model_name: str, model_path: str, raw_names) -> None:
    global class_names, class_translations, colors

    names_dict = normalize_model_names(raw_names)
    if not names_dict:
        names_dict = {idx: name for idx, name in enumerate(DEFAULT_CLASS_CATALOG)}

    max_id = max(names_dict.keys()) if names_dict else -1
    class_names = [names_dict.get(i, f"class_{i}") for i in range(max_id + 1)] if max_id >= 0 else []
    class_translations = {name: humanize_class_name(name) for name in class_names}
    colors = [DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)] for i in range(len(class_names))]

    ordered_ids = sorted(names_dict.keys())
    id_to_display = {cid: class_translations.get(names_dict[cid], humanize_class_name(names_dict[cid])) for cid in ordered_ids}
    id_to_color = {cid: colors[cid] if cid < len(colors) else DEFAULT_COLOR_PALETTE[cid % len(DEFAULT_COLOR_PALETTE)] for cid in ordered_ids}
    id_to_dataset_id = {cid: canonical_class_id_from_name(names_dict[cid]) for cid in ordered_ids}

    st.session_state.active_model_classes = {
        "model_name": model_name,
        "model_path": model_path,
        "id_to_name": names_dict,
        "id_to_display": id_to_display,
        "id_to_color": id_to_color,
        "id_to_dataset_id": id_to_dataset_id,
        "ordered_ids": ordered_ids,
        "non_final_classes": [names_dict[cid] for cid in ordered_ids if id_to_dataset_id.get(cid) is None],
        "payload": {
            "model_name": model_name,
            "model_path": model_path,
            "classes": [
                {
                    "id": cid,
                    "name": names_dict[cid],
                    "dataset_id": id_to_dataset_id.get(cid),
                    "dataset_name": get_dataset_class_name(id_to_dataset_id[cid]) if id_to_dataset_id.get(cid) is not None else None,
                }
                for cid in ordered_ids
            ]
        }
    }

    final_ids = sorted(FINAL_DATASET_CLASSES.keys())
    if "active_brush_class" not in st.session_state or st.session_state.active_brush_class not in final_ids:
        st.session_state.active_brush_class = final_ids[0]


def get_active_class_name(class_id: int) -> str:
    data = st.session_state.get("active_model_classes", {})
    return data.get("id_to_name", {}).get(class_id, f"class_{class_id}")


def get_active_class_display(class_id: int) -> str:
    data = st.session_state.get("active_model_classes", {})
    return data.get("id_to_display", {}).get(class_id, humanize_class_name(get_active_class_name(class_id)))


def get_active_class_color(class_id: int) -> str:
    data = st.session_state.get("active_model_classes", {})
    if "id_to_color" in data and class_id in data["id_to_color"]:
        return data["id_to_color"][class_id]
    return DEFAULT_COLOR_PALETTE[class_id % len(DEFAULT_COLOR_PALETTE)]


def ensure_annotation_session_dirs() -> dict:
    project_root = os.path.dirname(os.path.abspath(__file__))
    sessions_root = os.path.join(project_root, "annotations_sessions")
    os.makedirs(sessions_root, exist_ok=True)

    if "annotation_session_completed_dir" not in st.session_state:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        session_root = os.path.join(sessions_root, session_name)
        completed_dir = os.path.join(session_root, "completed")
        st.session_state.annotation_session_name = session_name
    else:
        completed_dir = st.session_state.annotation_session_completed_dir
        session_root = os.path.dirname(completed_dir)
        if not os.path.exists(session_root):
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            session_root = os.path.join(sessions_root, session_name)
            completed_dir = os.path.join(session_root, "completed")
            st.session_state.annotation_session_name = session_name

    os.makedirs(completed_dir, exist_ok=True)
    for sub in ["images", "labels", "metadata", "masks", "labels_segmentation"]:
        os.makedirs(os.path.join(completed_dir, sub), exist_ok=True)

    st.session_state.annotation_session_completed_dir = completed_dir
    return {
        "root": completed_dir,
        "images": os.path.join(completed_dir, "images"),
        "labels": os.path.join(completed_dir, "labels"),
        "metadata": os.path.join(completed_dir, "metadata"),
        "masks": os.path.join(completed_dir, "masks"),
        "labels_segmentation": os.path.join(completed_dir, "labels_segmentation"),
    }


def persist_annotation(filename: str, boxes: list, image_w: int, image_h: int, *, image_bgr=None, image_path=None, source: str = "tab1") -> dict:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data7_dir = os.path.join(project_root, "Data7.off")
    images_dir = os.path.join(data7_dir, "images")
    labels_dir = os.path.join(data7_dir, "labels")
    masks_dir = os.path.join(data7_dir, "masks")
    segmentation_labels_dir = os.path.join(data7_dir, "labels_segmentation")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(segmentation_labels_dir, exist_ok=True)

    filename_no_ext, _ = os.path.splitext(filename)
    img_save_path = os.path.join(images_dir, filename)
    txt_save_path = os.path.join(labels_dir, f"{filename_no_ext}.txt")

    if image_bgr is not None:
        cv2.imwrite(img_save_path, image_bgr)
    elif image_path:
        if os.path.abspath(image_path) != os.path.abspath(img_save_path):
            shutil.copy(image_path, img_save_path)
    else:
        raise ValueError("Either image_bgr or image_path must be provided to persist annotations.")

    metadata_boxes = []
    mask_paths = []
    segmentation_lines = []
    with open(txt_save_path, "w", encoding="utf-8") as f:
        for box_index, item in enumerate(boxes):
            cls_id = int(item['class_id'])
            x1, y1, x2, y2 = item['box']

            box_w = max(0.0, min(1.0, (x2 - x1) / image_w)) if image_w else 0.0
            box_h = max(0.0, min(1.0, (y2 - y1) / image_h)) if image_h else 0.0
            x_center = max(0.0, min(1.0, ((x1 + x2) / 2) / image_w)) if image_w else 0.0
            y_center = max(0.0, min(1.0, ((y1 + y2) / 2) / image_h)) if image_h else 0.0

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

            box_metadata = {
                "class_id": cls_id,
                "class_name": get_dataset_class_name(cls_id),
                "class_display": get_dataset_class_display(cls_id),
                "source_model_class_id": item.get("model_class_id"),
                "source_model_class_name": item.get("model_class_name"),
                "confidence": float(item.get('conf', 1.0)),
                "bbox_pixels": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "bbox_yolo": {
                    "x_center": round(x_center, 6),
                    "y_center": round(y_center, 6),
                    "width": round(box_w, 6),
                    "height": round(box_h, 6)
                }
            }
            mask_polygon = item.get("mask_polygon")
            if mask_polygon and len(mask_polygon) >= 3:
                normalized_polygon = []
                for point in mask_polygon:
                    px, py = float(point[0]), float(point[1])
                    normalized_polygon.extend([
                        max(0.0, min(1.0, px / image_w)) if image_w else 0.0,
                        max(0.0, min(1.0, py / image_h)) if image_h else 0.0,
                    ])
                segmentation_lines.append(
                    f"{cls_id} " + " ".join(f"{value:.6f}" for value in normalized_polygon)
                )
                mask_filename = f"{filename_no_ext}_{box_index:03d}.png"
                mask_path = os.path.join(masks_dir, mask_filename)
                binary_mask = np.zeros((image_h, image_w), dtype=np.uint8)
                cv2.fillPoly(
                    binary_mask,
                    [np.asarray(mask_polygon, dtype=np.int32)],
                    255,
                )
                cv2.imwrite(mask_path, binary_mask)
                mask_paths.append(mask_path)
                box_metadata["mask_polygon_pixels"] = mask_polygon
                box_metadata["mask_confidence"] = item.get("mask_confidence")
                box_metadata["mask_path"] = mask_path
            metadata_boxes.append(box_metadata)

    segmentation_label_path = os.path.join(
        segmentation_labels_dir, f"{filename_no_ext}.txt"
    )
    if segmentation_lines:
        with open(segmentation_label_path, "w", encoding="utf-8") as segmentation_file:
            segmentation_file.write("\n".join(segmentation_lines) + "\n")
    elif os.path.exists(segmentation_label_path):
        os.remove(segmentation_label_path)

    session_dirs = ensure_annotation_session_dirs()
    session_img_path = os.path.join(session_dirs["images"], filename)
    session_label_path = os.path.join(session_dirs["labels"], f"{filename_no_ext}.txt")
    metadata_path = os.path.join(session_dirs["metadata"], f"{filename_no_ext}.json")

    shutil.copy(img_save_path, session_img_path)
    shutil.copy(txt_save_path, session_label_path)
    for mask_path in mask_paths:
        shutil.copy(mask_path, os.path.join(session_dirs["masks"], os.path.basename(mask_path)))
    if segmentation_lines:
        shutil.copy(
            segmentation_label_path,
            os.path.join(session_dirs["labels_segmentation"], os.path.basename(segmentation_label_path)),
        )

    active_payload = st.session_state.get("active_model_classes", {}).get("payload", {})
    non_final_classes = st.session_state.get("active_model_classes", {}).get("non_final_classes", [])
    manual_flag = any(float(box.get('conf', 0.0)) >= 0.999 for box in metadata_boxes)

    metadata = {
        "filename": filename,
        "annotation_timestamp": datetime.now().isoformat(),
        "annotation_source": source,
        "annotation_model_name": active_payload.get("model_name"),
        "annotation_model_path": active_payload.get("model_path"),
        "annotation_model_classes": active_payload.get("classes", []),
        "annotation_boxes_count": len(metadata_boxes),
        "annotation_masks_count": len(mask_paths),
        "annotation_contains_manual_boxes": manual_flag,
        "requires_final_harmonization": False,
        "non_final_classes_present": [],
        "final_dataset_classes": FINAL_DATASET_CLASSES,
        "dataset_class_standard": "best_2 canonical: 0=crack, 1=dent, 2=glass shatter, 3=lamp broken, 4=scratch, 5=tire flat",
        "boxes": metadata_boxes,
        "session_name": st.session_state.get("annotation_session_name"),
    }

    with open(metadata_path, "w", encoding="utf-8") as fm:
        json.dump(metadata, fm, indent=2, ensure_ascii=False)

    if "validated_annotations" not in st.session_state:
        st.session_state.validated_annotations = {}
    st.session_state.validated_annotations[filename] = {
        "metadata_path": metadata_path,
        "session_name": st.session_state.get("annotation_session_name")
    }

    split_sync_info = sync_existing_split_item(project_root, filename_no_ext)

    return {
        "image_path": img_save_path,
        "label_path": txt_save_path,
        "split_sync": split_sync_info,
        "session_image_path": session_img_path,
        "metadata_path": metadata_path,
        "session_name": st.session_state.get("annotation_session_name"),
    }

# Patch Ultralytics cache scanning before any training/evaluation call on Windows.
patch_ultralytics_cache_pool_for_windows()

# Set page layout and style as the VERY FIRST Streamlit command
st.set_page_config(
    page_title="Car Damage Annotation Studio",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


def require_backoffice_login() -> None:
    if not BACKOFFICE_PASSWORD:
        return
    if st.session_state.get("backoffice_authenticated"):
        with st.sidebar:
            st.caption(f"Connecté : {BACKOFFICE_USERNAME}")
            if st.button("Se déconnecter"):
                st.session_state.pop("backoffice_authenticated", None)
                st.rerun()
        return

    st.title("Car Damage Detection - Backoffice")
    st.caption("Accès réservé aux personnes autorisées.")
    with st.form("backoffice_login"):
        username = st.text_input("Utilisateur", value=BACKOFFICE_USERNAME)
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter", type="primary")
    if submitted:
        if username == BACKOFFICE_USERNAME and password == BACKOFFICE_PASSWORD:
            st.session_state["backoffice_authenticated"] = True
            st.rerun()
        st.error("Identifiants invalides.")
    st.stop()


require_backoffice_login()

# Monkey-patch image_to_url for streamlit-drawable-canvas compatibility in modern Streamlit versions
try:
    import streamlit.elements.image as st_image
    from streamlit.elements.lib.image_utils import image_to_url
    
    class FakeLayoutConfig:
        def __init__(self, width):
            self.width = width

    original_image_to_url = image_to_url

    def wrapped_image_to_url(image_data, width, *args, **kwargs):
        # If the second argument is an integer, wrap it in our FakeLayoutConfig class
        if isinstance(width, int):
            layout_config = FakeLayoutConfig(width)
        else:
            layout_config = width
        return original_image_to_url(image_data, layout_config, *args, **kwargs)

    st_image.image_to_url = wrapped_image_to_url
except Exception:
    pass

from streamlit_drawable_canvas import st_canvas

# Custom CSS for modern premium dashboard styling
st.markdown("""
    <style>
    /* Dark Futuristic Theme */
    .stApp {
        background: linear-gradient(135deg, #090B11 0%, #0F131E 100%);
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp,
    .stApp p,
    .stApp li,
    .stApp span,
    .stApp label,
    .stApp div[data-testid="stMarkdownContainer"],
    .stApp div[data-testid="stMarkdownContainer"] p {
        color: #F8FAFC !important;
    }

    .stApp small,
    .stCaptionContainer,
    .stCaptionContainer p,
    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p {
        color: #D7E2F1 !important;
    }

    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-baseweb="select"] *,
    div[data-baseweb="popover"] *,
    div[role="listbox"] *,
    div[role="option"] *,
    ul[role="listbox"] *,
    div[data-testid="stFileUploader"] *,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] p,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label,
    input,
    textarea {
        color: #FFFFFF !important;
    }

    div[data-baseweb="popover"],
    div[role="listbox"],
    ul[role="listbox"],
    div[data-baseweb="menu"] {
        background-color: #0B0E17 !important;
    }

    div[role="option"]:hover,
    ul[role="listbox"] li:hover {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        font-weight: 800 !important;
        letter-spacing: 0;
        color: #FFFFFF !important;
    }
    
    .main-title {
        background: linear-gradient(90deg, #FF4B4B 0%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        margin-bottom: 5px !important;
        padding-bottom: 10px;
    }
    
    /* Sidebar premium redesign */
    [data-testid="stSidebar"] {
        background-color: #0B0E17 !important;
        border-right: 1px solid #1E293B;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small {
        color: #FFFFFF !important;
    }
    
    /* Sleek container cards */
    .premium-card {
        background-color: #131926;
        color: #FFFFFF !important;
        border: 1px solid #3B4A63;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .box-card {
        background-color: #161D2E;
        color: #FFFFFF !important;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid #222C43;
        transition: transform 0.2s, border-color 0.2s;
    }
    .box-card:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
    }
    
    /* Metrics panel styling */
    .metric-value {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
    }
    
    /* Glowing status badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* Modern buttons design */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px;
        transition: all 0.2s ease-in-out !important;
        background-color: #05070D !important;
        color: #FFFFFF !important;
        border: 1px solid #64748B !important;
        min-height: 44px;
    }

    div.stButton > button p,
    div.stButton > button span {
        color: inherit !important;
    }

    div.stButton > button[data-testid="baseButton-primary"],
    div.stButton > button[kind="primary"] {
        background-color: #FFFFFF !important;
        color: #05070D !important;
        border: 2px solid #38BDF8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.18) !important;
    }

    div.stButton > button[data-testid="baseButton-primary"] *,
    div.stButton > button[kind="primary"] *,
    button[data-testid="baseButton-primary"] *,
    button[kind="primary"] * {
        color: #05070D !important;
    }

    div.stButton > button[data-testid="baseButton-secondary"],
    div.stButton > button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #05070D !important;
        border: 2px solid #CBD5E1 !important;
    }

    div.stButton > button[data-testid="baseButton-secondary"] *,
    div.stButton > button[kind="secondary"] *,
    button[data-testid="baseButton-secondary"] *,
    button[kind="secondary"] * {
        color: #05070D !important;
    }

    div.stButton > button[data-testid="baseButton-secondary"]:hover,
    div.stButton > button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[kind="secondary"]:hover {
        background-color: #E2E8F0 !important;
        color: #020617 !important;
        border-color: #38BDF8 !important;
    }

    div.stButton > button[data-testid="baseButton-secondary"]:hover *,
    div.stButton > button[kind="secondary"]:hover *,
    button[data-testid="baseButton-secondary"]:hover *,
    button[kind="secondary"]:hover * {
        color: #020617 !important;
    }

    input,
    textarea,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background-color: #FFFFFF !important;
        color: #05070D !important;
        caret-color: #05070D !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #475569 !important;
        opacity: 1 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #05070D !important;
    }

    div[data-baseweb="select"] > div *,
    div[data-baseweb="popover"] * {
        color: #05070D !important;
    }
    
    div.stButton > button:hover {
        background-color: #111827 !important;
        color: #FFFFFF !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.25) !important;
    }

    div.stButton > button[data-testid="baseButton-primary"]:hover,
    div.stButton > button[kind="primary"]:hover {
        background-color: #F8FAFC !important;
        color: #020617 !important;
    }

    /* Final override: every Streamlit button/square must keep dark text on light background. */
    div.stButton button,
    div.stButton button:hover,
    div.stButton button:focus,
    div.stButton button:active,
    div[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploader"] button:hover,
    div[data-testid="stFileUploader"] button:focus,
    div[data-testid="stFileUploader"] button:active {
        background-color: #FFFFFF !important;
        color: #05070D !important;
        border: 2px solid #CBD5E1 !important;
    }

    div.stButton button *,
    div.stButton button:hover *,
    div.stButton button:focus *,
    div.stButton button:active *,
    div[data-testid="stFileUploader"] button *,
    div[data-testid="stFileUploader"] button:hover *,
    div[data-testid="stFileUploader"] button:focus *,
    div[data-testid="stFileUploader"] button:active * {
        color: #05070D !important;
        -webkit-text-fill-color: #05070D !important;
    }
    
    /* Instructions card style */
    .instruction-step {
        background-color: #131926;
        border-left: 4px solid #FF4B4B;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }

    .app-subtitle {
        font-size: 1.08rem;
        color: #F1F5F9 !important;
        margin-top: -10px;
        margin-bottom: 22px;
        max-width: 980px;
        line-height: 1.55;
    }

    .nav-kicker {
        color: #BAE6FD !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.78rem;
        margin-bottom: 4px;
    }

    .nav-title {
        color: #FFFFFF !important;
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .nav-caption {
        color: #E2E8F0 !important;
        font-size: 0.95rem;
        margin-bottom: 12px;
    }

    .nav-core {
        width: 116px;
        height: 116px;
        border-radius: 999px;
        margin: 8px auto 0 auto;
        background: radial-gradient(circle at 35% 30%, #1F293D 0%, #101827 62%, #0B1020 100%);
        border: 1px solid #38BDF8;
        box-shadow: 0 0 22px rgba(56, 189, 248, 0.20);
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #E0F2FE;
        font-weight: 800;
        line-height: 1.25;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("<h1 class='main-title'>🚗 Atelier dommages auto</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='app-subtitle'>Détectez les dommages sur une photo, corrigez les boîtes à la main, préparez le dataset, puis suivez les performances du modèle.</p>",
    unsafe_allow_html=True,
)

# Base Directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Model Loader
@st.cache_resource
def get_cached_yolo_model(resolved_path, mtime):
    """
    Loads and caches the YOLO model. If the file modification time (mtime) changes,
    Streamlit will invalidate the cache for this resolved_path and reload the model.
    """
    return YOLO(resolved_path)

def load_model(model_path):
    try:
        abs_model_path = os.path.join(base_dir, model_path)
        resolved_path = None
        is_custom = False
        
        if os.path.exists(abs_model_path):
            resolved_path = abs_model_path
            is_custom = True
        elif os.path.exists(model_path):
            resolved_path = os.path.abspath(model_path)
            is_custom = True
        else:
            if os.path.basename(model_path) == "best_2.pt":
                st.sidebar.error("Le modèle de référence `best_2.pt` est introuvable.")
                return None, False
            resolved_path = "best_2.pt"
            is_custom = False
            
        # Get modification time to automatically invalidate the cache if the file changes
        mtime = 0.0
        if is_custom and resolved_path and os.path.exists(resolved_path):
            mtime = os.path.getmtime(resolved_path)
            
        # Load cached model
        loaded_model = get_cached_yolo_model(resolved_path, mtime)
        return loaded_model, is_custom
    except Exception as e:
        st.sidebar.error(f"Erreur de chargement : {e}")
        return None, False


def is_cuda_runtime_error(error: Exception) -> bool:
    message = str(error).lower()
    markers = (
        "cuda",
        "cudnn",
        "cublas",
        "device-side assert",
        "out of memory",
        "driver shutting down",
    )
    return any(marker in message for marker in markers)


def run_inference_safely(
    loaded_model,
    image_source,
    confidence: float,
    iou: float,
    model_path: str,
    prefer_gpu: bool = True,
):
    requested_device = 0 if prefer_gpu and cuda_is_available() else "cpu"
    try:
        result = loaded_model.predict(
            source=image_source,
            conf=confidence,
            iou=iou,
            verbose=False,
            device=requested_device,
        )[0]
        return result, ("GPU CUDA" if requested_device == 0 else "CPU"), None
    except Exception as gpu_error:
        if requested_device != 0 or not is_cuda_runtime_error(gpu_error):
            raise
        try:
            import torch

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            resolved_path = model_path
            if not os.path.isabs(resolved_path):
                local_path = os.path.join(APP_BASE_DIR, resolved_path)
                if os.path.exists(local_path):
                    resolved_path = local_path
            cpu_model = YOLO(resolved_path)
            result = cpu_model.predict(
                source=image_source,
                conf=confidence,
                iou=iou,
                verbose=False,
                device="cpu",
            )[0]
            return result, "CPU (secours apres erreur CUDA)", str(gpu_error)
        except Exception as cpu_error:
            raise RuntimeError(
                f"CUDA a echoue ({gpu_error}) puis le secours CPU a echoue ({cpu_error})."
            ) from cpu_error


@st.cache_resource(show_spinner=False)
def get_cached_sam2_predictor(
    package_root: str,
    checkpoint_path: str,
    config_name: str,
    device: str,
):
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(config_name, checkpoint_path, device=device)
    return SAM2ImagePredictor(sam2_model)


def load_video_sam2_predictor():
    if not os.path.isdir(SAM2_PACKAGE_ROOT):
        return None, f"Package SAM2 introuvable: {SAM2_PACKAGE_ROOT}"
    if not os.path.isfile(SAM2_CHECKPOINT_PATH):
        return None, f"Checkpoint SAM2 introuvable: {SAM2_CHECKPOINT_PATH}"
    try:
        requested_device = os.getenv("SAM2_DEVICE", "cpu").lower()
        device = "cuda" if requested_device == "cuda" and cuda_is_available() else "cpu"
        predictor = get_cached_sam2_predictor(
            SAM2_PACKAGE_ROOT,
            SAM2_CHECKPOINT_PATH,
            SAM2_CONFIG_NAME,
            device,
        )
        return predictor, f"SAM2 reel actif sur {device.upper()}"
    except Exception as error:
        return None, f"SAM2 indisponible: {error}"


def segment_bbox_with_sam2(predictor, frame_bgr: np.ndarray, bbox: list[float]):
    predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(
        box=np.asarray(bbox, dtype=np.float32),
        multimask_output=True,
    )
    best_index = int(np.argmax(scores))
    binary = masks[best_index].astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("SAM2 a retourne un masque vide")
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).tolist()
    if len(polygon) < 3:
        raise RuntimeError("Le masque SAM2 ne contient pas assez de points")
    return polygon, float(scores[best_index])


def apply_sam2_to_boxes(
    image_bgr: np.ndarray,
    boxes: list,
    predictor,
    *,
    max_boxes: int | None = None,
    only_missing: bool = True,
):
    updated_boxes = [dict(item) for item in boxes]
    errors = []
    processed = 0
    limit = len(updated_boxes) if max_boxes is None else max(0, int(max_boxes))

    for idx, item in enumerate(updated_boxes):
        if processed >= limit:
            break
        if only_missing and item.get("mask_polygon"):
            continue
        try:
            polygon, mask_confidence = segment_bbox_with_sam2(
                predictor, image_bgr, item["box"]
            )
            updated_boxes[idx]["mask_polygon"] = polygon
            updated_boxes[idx]["mask_confidence"] = mask_confidence
            processed += 1
        except Exception as error:
            errors.append(f"Boîte #{idx + 1}: {error}")

    return updated_boxes, {"processed": processed, "errors": errors}


def video_frame_quality_score(frame_bgr: np.ndarray, confidence: float, bbox: list[float]) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 250.0)
    brightness_quality = max(0.0, 1.0 - abs(float(gray.mean()) - 128.0) / 128.0)
    reflection_ratio = float((gray >= 235).mean())
    reflection_quality = max(0.0, 1.0 - reflection_ratio / 0.12)
    x1, y1, x2, y2 = bbox
    edge_margin = min(x1, y1, width - x2, height - y2)
    edge_quality = min(1.0, max(0.0, edge_margin) / max(1.0, min(width, height) * 0.04))
    return round(
        0.62 * float(confidence)
        + 0.18 * sharpness
        + 0.08 * brightness_quality
        + 0.07 * reflection_quality
        + 0.05 * edge_quality,
        6,
    )


def analyze_video_with_bytetrack(
    video_bytes: bytes,
    original_name: str,
    loaded_model,
    model_path: str,
    model_to_dataset_id: dict,
    id_to_model_name: dict,
    confidence: float,
    iou: float,
    frame_stride: int,
    sampling_seconds: float | None,
    max_samples: int,
    prefer_gpu: bool,
    sam2_predictor=None,
    max_sam2_tracks: int = 8,
    progress_callback=None,
) -> dict:
    started_at = time.perf_counter()

    def report_progress(percent: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), message)

    report_progress(0, "Préparation de la vidéo")
    suffix = os.path.splitext(original_name)[1].lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        suffix = ".mp4"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
            temp_video.write(video_bytes)
            temp_path = temp_video.name

        capture = cv2.VideoCapture(temp_path)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        effective_stride = (
            max(1, int(round(fps * sampling_seconds)))
            if sampling_seconds is not None and sampling_seconds > 0
            else max(1, int(frame_stride))
        )
        expected_samples = (
            min(max_samples, max(1, (total_frames + effective_stride - 1) // effective_stride))
            if total_frames else max_samples
        )

        tracks: dict[str, dict] = {}
        ignored_predictions = defaultdict(int)
        sampled_frames = 0
        fallback_track_id = 100000

        def consume_stream(tracking_model, device):
            nonlocal sampled_frames, fallback_track_id
            stream = tracking_model.track(
                source=temp_path,
                stream=True,
                persist=False,
                tracker="bytetrack.yaml",
                conf=confidence,
                iou=iou,
                vid_stride=effective_stride,
                verbose=False,
                device=device,
                classes=sorted(model_to_dataset_id) or None,
            )
            try:
                for sample_index, result in enumerate(stream):
                    if sample_index >= max_samples:
                        break
                    sampled_frames += 1
                    report_progress(
                        round(70 * sampled_frames / max(1, expected_samples)),
                        f"YOLO + ByteTrack : frame {sampled_frames}/{expected_samples}",
                    )
                    frame_bgr = result.orig_img
                    frame_index = min(
                        max(0, total_frames - 1), sample_index * effective_stride
                    ) if total_frames else sample_index * effective_stride
                    if result.boxes is None:
                        continue
                    for detection_index, box in enumerate(result.boxes):
                        model_class_id = int(box.cls[0].item())
                        class_id = model_to_dataset_id.get(model_class_id)
                        model_class_name = id_to_model_name.get(
                            model_class_id, f"class_{model_class_id}"
                        )
                        if class_id is None:
                            ignored_predictions[model_class_name] += 1
                            continue
                        bbox = [float(value) for value in box.xyxy[0].tolist()]
                        confidence_value = float(box.conf[0].item())
                        box_track_id = getattr(box, "id", None)
                        if box_track_id is None:
                            raw_track_id = fallback_track_id
                            fallback_track_id += 1
                        else:
                            raw_track_id = int(box_track_id[0].item())
                        track_key = f"{class_id}:{raw_track_id}"
                        track = tracks.setdefault(
                            track_key,
                            {
                                "track_id": f"{get_dataset_class_name(class_id)}-{raw_track_id:03d}",
                                "class_id": class_id,
                                "model_class_id": model_class_id,
                                "model_class_name": model_class_name,
                                "frames_seen": 0,
                                "best_score": -1.0,
                            },
                        )
                        track["frames_seen"] += 1
                        quality_score = video_frame_quality_score(
                            frame_bgr, confidence_value, bbox
                        )
                        if quality_score > track["best_score"]:
                            ok, encoded = cv2.imencode(
                                ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94]
                            )
                            if not ok:
                                continue
                            track.update(
                                {
                                    "best_score": quality_score,
                                    "confidence": confidence_value,
                                    "bbox": bbox,
                                    "frame_index": frame_index,
                                    "timestamp_seconds": frame_index / fps,
                                    "frame_jpeg": encoded.tobytes(),
                                }
                            )
            finally:
                close_stream = getattr(stream, "close", None)
                if callable(close_stream):
                    close_stream()

        requested_device = 0 if prefer_gpu and cuda_is_available() else "cpu"
        used_device = "GPU CUDA" if requested_device == 0 else "CPU"
        cuda_fallback_error = None
        tracking_started_at = time.perf_counter()
        try:
            consume_stream(loaded_model, requested_device)
        except Exception as gpu_error:
            if requested_device != 0 or not is_cuda_runtime_error(gpu_error):
                raise
            cuda_fallback_error = str(gpu_error)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            tracks.clear()
            sampled_frames = 0
            resolved_path = model_path
            if not os.path.isabs(resolved_path):
                local_path = os.path.join(APP_BASE_DIR, resolved_path)
                if os.path.exists(local_path):
                    resolved_path = local_path
            consume_stream(YOLO(resolved_path), "cpu")
            used_device = "CPU (secours apres erreur CUDA)"
        tracking_elapsed = time.perf_counter() - tracking_started_at

        sam2_errors = []
        ordered_tracks = sorted(
            tracks.values(),
            key=lambda item: (-item["best_score"], item["track_id"]),
        )
        sam2_started_at = time.perf_counter()
        if sam2_predictor is not None:
            tracks_to_segment = ordered_tracks[:max(0, int(max_sam2_tracks))]
            for track_index, track in enumerate(tracks_to_segment, start=1):
                report_progress(
                    70 + round(30 * (track_index - 1) / max(1, len(tracks_to_segment))),
                    f"SAM2 : piste {track_index}/{len(tracks_to_segment)}",
                )
                frame_array = np.frombuffer(track["frame_jpeg"], np.uint8)
                frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                try:
                    polygon, mask_confidence = segment_bbox_with_sam2(
                        sam2_predictor, frame_bgr, track["bbox"]
                    )
                    track["mask_polygon"] = polygon
                    track["mask_confidence"] = mask_confidence
                except Exception as error:
                    track["mask_polygon"] = None
                    track["mask_confidence"] = None
                    sam2_errors.append(f"{track['track_id']}: {error}")
        sam2_elapsed = time.perf_counter() - sam2_started_at
        report_progress(100, "Analyse vidéo terminée")
        return {
            "video_name": original_name,
            "fps": fps,
            "total_frames": total_frames,
            "sampled_frames": sampled_frames,
            "frame_stride": effective_stride,
            "tracks": ordered_tracks,
            "ignored_predictions": dict(ignored_predictions),
            "used_device": used_device,
            "cuda_fallback_error": cuda_fallback_error,
            "sam2_errors": sam2_errors,
            "tracking_elapsed_seconds": round(tracking_elapsed, 2),
            "sam2_elapsed_seconds": round(sam2_elapsed, 2),
            "total_elapsed_seconds": round(time.perf_counter() - started_at, 2),
            "sam2_tracks_processed": min(
                len(ordered_tracks), max(0, int(max_sam2_tracks))
            ) if sam2_predictor is not None else 0,
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def render_video_track_preview(frame_bgr: np.ndarray, track: dict) -> np.ndarray:
    preview = frame_bgr.copy()
    color_hex = get_dataset_class_color(track["class_id"]).lstrip("#")
    rgb_color = tuple(int(color_hex[index:index + 2], 16) for index in (0, 2, 4))
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    polygon = track.get("mask_polygon")
    if polygon and len(polygon) >= 3:
        points = np.asarray(polygon, dtype=np.int32)
        overlay = preview.copy()
        cv2.fillPoly(overlay, [points], bgr_color)
        preview = cv2.addWeighted(overlay, 0.28, preview, 0.72, 0)
        cv2.polylines(preview, [points], True, bgr_color, 3, cv2.LINE_AA)
    x1, y1, x2, y2 = [int(value) for value in track["bbox"]]
    cv2.rectangle(preview, (x1, y1), (x2, y2), bgr_color, 3)
    label = f"{track['track_id']} | {track['confidence']:.0%}"
    cv2.putText(
        preview, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, bgr_color, 2, cv2.LINE_AA,
    )
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

# Sidebar Design
st.sidebar.markdown("<h2 style='text-align: center; color: #FFF; margin-bottom: 10px;'>🧠 Contrôles IA</h2>", unsafe_allow_html=True)

# Model Selector in Sidebar
st.sidebar.subheader("🤖 Modèle utilisé")

available_models = {
    "Référence unique (best_2.pt)": "best_2.pt"
}
for pt_file in sorted([f for f in os.listdir(base_dir) if f.lower().endswith(".pt") and f.startswith("data7_from_best_2")]):
    available_models[f"Modèle Data7 ({pt_file})"] = pt_file

# Check which models actually exist
existing_models = {k: v for k, v in available_models.items() if os.path.exists(os.path.join(base_dir, v))}
if not existing_models:
    st.sidebar.error("`best_2.pt` est introuvable. Place ce fichier dans le dossier du projet pour utiliser l'application.")
    existing_models = {"Référence unique (best_2.pt manquant)": "best_2.pt"}
model_options = list(existing_models.keys())

MODEL_SELECTOR_KEY = "model_selector"
if "active_model_name" not in st.session_state or st.session_state.active_model_name not in model_options:
    st.session_state.active_model_name = model_options[0]
    st.session_state.active_model_path = existing_models[st.session_state.active_model_name]

if st.session_state.get("pending_model_synced"):
    st.session_state[MODEL_SELECTOR_KEY] = st.session_state.pending_model_synced
    st.session_state.pending_model_synced = None

if MODEL_SELECTOR_KEY not in st.session_state or st.session_state[MODEL_SELECTOR_KEY] not in model_options:
    st.session_state[MODEL_SELECTOR_KEY] = st.session_state.active_model_name

if "model_change_candidate" not in st.session_state:
    st.session_state.model_change_candidate = None
if "pending_model_reload" not in st.session_state:
    st.session_state.pending_model_reload = False
if "pending_model_synced" not in st.session_state:
    st.session_state.pending_model_synced = None

selected_model_name = st.sidebar.selectbox(
    "Choisir le modèle :",
    model_options,
    key=MODEL_SELECTOR_KEY,
    help="Ce modèle sert à détecter automatiquement les dommages sur les images importées."
)

if selected_model_name != st.session_state.active_model_name:
    st.session_state.model_change_candidate = selected_model_name

if st.session_state.model_change_candidate:
    candidate_name = st.session_state.model_change_candidate
    candidate_path = existing_models.get(candidate_name)
    st.sidebar.warning(
        "Vous changez de modèle. Les prédictions actuellement affichées ont été générées par l'ancien modèle. "
        "Voulez-vous relancer l'analyse avec le nouveau modèle ?"
    )
    col_confirm, col_cancel = st.sidebar.columns(2)
    with col_confirm:
        if st.button("Relancer avec le nouveau modèle", key="confirm_model_change", use_container_width=True):
            st.session_state.active_model_name = candidate_name
            st.session_state.active_model_path = candidate_path
            st.session_state.model_change_candidate = None
            st.session_state.pending_model_reload = True
            st.session_state.pending_model_synced = candidate_name
            st.rerun()
    with col_cancel:
        if st.button("Annuler le changement", key="cancel_model_change", use_container_width=True):
            st.session_state.pending_model_synced = st.session_state.active_model_name
            st.session_state.model_change_candidate = None
            st.rerun()

active_model_name = st.session_state.active_model_name
active_model_path = st.session_state.active_model_path

# Load the selected model
model, is_custom = load_model(active_model_path)

# Synchronize active classes with the loaded model
if model is not None:
    sync_active_model_classes(active_model_name, active_model_path, getattr(model, 'names', {}))
elif "active_model_classes" not in st.session_state:
    sync_active_model_classes(active_model_name, active_model_path, DEFAULT_CLASS_CATALOG)

if is_custom:
    st.sidebar.markdown(f"<div class='badge badge-success' style='width: 100%; text-align: center;'>✅ {active_model_name} Actif</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div class='badge badge-warning' style='width: 100%; text-align: center;'>⚠️ Mode Démo Actif</div>", unsafe_allow_html=True)

if st.session_state.get("pending_model_reload"):
    st.session_state.tab1_current_file = None
    st.session_state.tab1_current_image_key = None
    st.session_state.tab1_boxes = []
    st.session_state.tab1_history = []
    if "canvas_key" in st.session_state:
        st.session_state.canvas_key += 1
    st.sidebar.success(
        f"Pinceau synchronisé avec le modèle actif : {active_model_name} — {len(st.session_state.active_model_classes.get('ordered_ids', []))} classes disponibles."
    )
    st.session_state.pending_model_reload = False

# Active Drawing Class
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Classe à dessiner")

available_class_ids = order_annotation_class_ids(sorted(FINAL_DATASET_CLASSES.keys()))
if "active_brush_class" not in st.session_state or st.session_state.active_brush_class not in available_class_ids:
    st.session_state.active_brush_class = available_class_ids[0]

def _format_brush_option(class_id: int) -> str:
    return f"ID {class_id} — {get_dataset_class_display(class_id)}"

selected_class_id = st.sidebar.selectbox(
    "Type de dommage :",
    options=available_class_ids,
    format_func=_format_brush_option,
    key="active_brush_class"
)
active_color = get_dataset_class_color(selected_class_id)

with st.sidebar.expander("Voir les classes du modèle actif", expanded=False):
    st.markdown(f"**Modèle actif :** `{active_model_name}`")
    table_rows = "\n".join([
        f"| {cid} | {get_active_class_name(cid)} | {get_dataset_class_display(st.session_state.active_model_classes.get('id_to_dataset_id', {}).get(cid)) if st.session_state.active_model_classes.get('id_to_dataset_id', {}).get(cid) is not None else 'Ignorée' } |"
        for cid in st.session_state.active_model_classes.get("ordered_ids", [])
    ])
    st.markdown("| ID modèle | Classe modèle | Classe Data7 |\n|---:|---|---|\n" + table_rows)
    st.json(st.session_state.active_model_classes.get("payload", {}))

st.sidebar.markdown(
    f"<small style='color:#94A3B8;'>Classes synchronisées avec : <strong>{active_model_name}</strong></small>",
    unsafe_allow_html=True
)

non_final_classes = st.session_state.active_model_classes.get("non_final_classes", [])
if non_final_classes:
    st.sidebar.warning(
        "Certaines classes du modèle actif ne correspondent pas à `scratch`, `dent` ou `crack`; elles seront ignorées à l'annotation automatique : "
        + ", ".join(non_final_classes)
    )

# Settings Section
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Réglages de détection")
conf_threshold = st.sidebar.slider("Confiance minimale", 0.10, 1.00, 0.25, 0.05)
iou_threshold = st.sidebar.slider("Fusion des boîtes (IoU)", 0.10, 1.00, 0.45, 0.05)
inference_device_choice = st.sidebar.selectbox(
    "Accélérateur d'analyse",
    ["GPU CUDA (avec secours CPU)", "CPU uniquement"],
    help="Le mode GPU bascule automatiquement sur CPU si le contexte CUDA devient invalide.",
)
prefer_gpu_inference = inference_device_choice.startswith("GPU")
if prefer_gpu_inference and cuda_is_available():
    st.sidebar.caption("Analyse automatique sur NVIDIA CUDA.")
elif prefer_gpu_inference:
    st.sidebar.warning("CUDA indisponible : les analyses utiliseront le CPU.")

last_inference_device = st.session_state.get("last_inference_device")
if last_inference_device:
    st.sidebar.caption(f"Dernière analyse : {last_inference_device}")
if st.session_state.get("last_cuda_fallback_error"):
    st.sidebar.warning(
        "Une erreur CUDA a été contournée automatiquement. La dernière image a été analysée sur CPU."
    )

# Session State Initialization - ISOLATED FOR EACH TAB TO AVOID INTERFERENCE
if "tab1_boxes" not in st.session_state:
    st.session_state.tab1_boxes = []
if "tab1_current_file" not in st.session_state:
    st.session_state.tab1_current_file = None
if "tab1_current_image_key" not in st.session_state:
    st.session_state.tab1_current_image_key = None
if "tab1_history" not in st.session_state:
    st.session_state.tab1_history = []

if "batch_images" not in st.session_state:
    st.session_state.batch_images = []
if "batch_index" not in st.session_state:
    st.session_state.batch_index = 0
if "video_analysis" not in st.session_state:
    st.session_state.video_analysis = None
if "video_upload_hash" not in st.session_state:
    st.session_state.video_upload_hash = None
if "video_editor_boxes" not in st.session_state:
    st.session_state.video_editor_boxes = []

if "tab2_boxes" not in st.session_state:
    st.session_state.tab2_boxes = []
if "tab2_current_file" not in st.session_state:
    st.session_state.tab2_current_file = None
if "tab2_history" not in st.session_state:
    st.session_state.tab2_history = []

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "tab1_loaded_existing_annotations" not in st.session_state:
    st.session_state.tab1_loaded_existing_annotations = {}

# Helper function to save current box state to history for Undo functionality
def save_history(tab_name):
    boxes_key = f"{tab_name}_boxes"
    hist_key = f"{tab_name}_history"
    # Make a deep copy of the current state of boxes and save it to history
    copied_boxes = [dict(b) for b in st.session_state[boxes_key]]
    st.session_state[hist_key].append(copied_boxes)
    # Limit history size to 20
    if len(st.session_state[hist_key]) > 20:
        st.session_state[hist_key].pop(0)

# Navigation. Streamlit tabs execute every tab on each rerun, so this custom
# selector keeps heavy sections lazy and prevents hidden model/dataset work.
NAV_SECTIONS = [
    "🚀 Scanner",
    "📁 Dataset",
    "🔄 Réentraîner",
    "📊 Performances",
    "🔍 Harmoniser",
]
NAV_HELP = {
    NAV_SECTIONS[0]: "Importer une photo ou un lot, lancer la détection, corriger les boîtes.",
    NAV_SECTIONS[1]: "Voir et modifier les images déjà sauvegardées dans Data7.off.",
    NAV_SECTIONS[2]: "Relancer l'entraînement du modèle avec les annotations validées.",
    NAV_SECTIONS[3]: "Évaluer un modèle et comparer les versions disponibles.",
    NAV_SECTIONS[4]: "Aligner les classes et générer le dataset final harmonisé.",
}

if "main_navigation" not in st.session_state or st.session_state.main_navigation not in NAV_SECTIONS:
    st.session_state.main_navigation = NAV_SECTIONS[0]
selected_section = st.session_state.main_navigation


def nav_button(section_name: str, label: str, key: str) -> None:
    btn_type = "primary" if selected_section == section_name else "secondary"
    if st.button(label, key=key, type=btn_type, use_container_width=True, help=NAV_HELP[section_name]):
        if st.session_state.main_navigation != section_name:
            st.session_state.main_navigation = section_name
            st.rerun()


st.markdown("<div class='nav-kicker'>Parcours rapide</div>", unsafe_allow_html=True)
st.markdown("<div class='nav-title'>Choisissez la zone de travail</div>", unsafe_allow_html=True)
st.markdown("<div class='nav-caption'>Chaque bouton ouvre une seule partie de l'application pour garder l'interface plus rapide.</div>", unsafe_allow_html=True)

nav_top = st.columns([1, 1.1, 1])
with nav_top[1]:
    nav_button(NAV_SECTIONS[0], "🚀 Scanner", "nav_scanner")

nav_mid = st.columns([1.15, 0.9, 1.15])
with nav_mid[0]:
    nav_button(NAV_SECTIONS[1], "📁 Dataset", "nav_dataset")
with nav_mid[1]:
    st.markdown("<div class='nav-core'>Flux<br>IA</div>", unsafe_allow_html=True)
with nav_mid[2]:
    nav_button(NAV_SECTIONS[3], "📊 Performances", "nav_performance")

nav_bottom = st.columns([1.15, 0.25, 1.15])
with nav_bottom[0]:
    nav_button(NAV_SECTIONS[2], "🔄 Réentraîner", "nav_training")
with nav_bottom[2]:
    nav_button(NAV_SECTIONS[4], "🔍 Harmoniser", "nav_harmonize")

st.markdown("---")

# --- TAB 1: SCAN & ANNOTATE ---
if selected_section == NAV_SECTIONS[0]:
    # Choose work mode
    st.markdown("<h3 style='color: #38BDF8; margin-top:0px;'>🎯 Importer et annoter</h3>", unsafe_allow_html=True)
    work_mode = st.radio(
        "Choisissez le type d'import :",
        ["📷 Une image", "🎥 Une vidéo", "📁 Plusieurs images ou ZIP"],
        horizontal=True,
        key="annotation_work_mode"
    )
    
    img_bgr = None
    filename = None
    orig_w, orig_h = 0, 0
    image_enable_sam2 = False
    image_max_sam2_boxes = 20
    
    if work_mode == "📷 Une image":
        uploaded_file = st.file_uploader("Déposez une photo de véhicule", type=["jpg", "jpeg", "png", "webp"], key="uploader_tab1")
        image_sam2_col, image_sam2_limit_col = st.columns(2)
        with image_sam2_col:
            image_enable_sam2 = st.checkbox(
                "Activer SAM2 sur l'image",
                value=True,
                key="image_enable_sam2",
                help="YOLO trouve les boîtes, puis SAM2 transforme ces boîtes en masques/polygones.",
            )
        with image_sam2_limit_col:
            image_max_sam2_boxes = st.number_input(
                "Boîtes maximum à segmenter",
                1,
                50,
                20,
                1,
                disabled=not image_enable_sam2,
                key="image_max_sam2_boxes",
            )
        if uploaded_file is not None:
            uploaded_file.seek(0)
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            filename = uploaded_file.name
            if img_bgr is None:
                st.error("Impossible de lire cette image. Essayez un fichier JPG, PNG ou WEBP valide.")
                filename = None
    elif work_mode == "🎥 Une vidéo":
        uploaded_video = st.file_uploader(
            "Déposez une vidéo du véhicule",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="uploader_video_tab1",
        )
        sampling_mode = st.radio(
            "Échantillonnage de la vidéo",
            ["Toutes les X secondes", "Toutes les X frames"],
            horizontal=True,
            key="video_sampling_mode",
        )
        sampling_col, limit_col = st.columns(2)
        with sampling_col:
            if sampling_mode == "Toutes les X secondes":
                video_sampling_seconds = st.number_input(
                    "Intervalle (secondes)", 0.1, 10.0, 1.0, 0.1,
                    key="video_sampling_seconds",
                )
                video_frame_stride = 1
            else:
                video_frame_stride = st.number_input(
                    "Intervalle (frames)", 1, 300, 10, 1,
                    key="video_frame_stride",
                )
                video_sampling_seconds = None
        with limit_col:
            video_max_samples = st.number_input(
                "Frames maximum à analyser", 10, 500, 120, 10,
                key="video_max_samples",
            )
        sam2_col, sam2_limit_col = st.columns(2)
        with sam2_col:
            video_enable_sam2 = st.checkbox(
                "Activer la segmentation SAM2",
                value=True,
                key="video_enable_sam2",
                help="Désactivez-la pour un résultat YOLO + ByteTrack beaucoup plus rapide.",
            )
        with sam2_limit_col:
            video_max_sam2_tracks = st.number_input(
                "Pistes maximum à segmenter",
                1,
                50,
                8,
                1,
                disabled=not video_enable_sam2,
                key="video_max_sam2_tracks",
            )

        if uploaded_video is not None:
            video_bytes = uploaded_video.getvalue()
            current_video_hash = hashlib.sha256(video_bytes).hexdigest()
            if st.session_state.video_upload_hash != current_video_hash:
                st.session_state.video_upload_hash = current_video_hash
                st.session_state.video_analysis = None
                st.session_state.video_editor_boxes = []
                st.session_state.video_editor_key = None
            st.video(video_bytes)

            if st.button(
                (
                    "Lancer YOLO + ByteTrack + SAM2"
                    if video_enable_sam2 else "Lancer YOLO + ByteTrack (rapide)"
                ),
                type="primary",
                use_container_width=True,
                disabled=model is None,
            ):
                try:
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def update_video_progress(percent: int, message: str) -> None:
                        progress_bar.progress(percent)
                        progress_text.markdown(f"**{percent}%** — {message}")

                    if video_enable_sam2:
                        update_video_progress(1, "Chargement du modèle SAM2")
                        sam2_predictor, sam2_status = load_video_sam2_predictor()
                    else:
                        sam2_predictor = None
                        sam2_status = "SAM2 désactivé : mode rapide YOLO + ByteTrack"
                    with st.spinner(
                        "Détection, suivi des défauts et segmentation des meilleures frames..."
                    ):
                        video_analysis = analyze_video_with_bytetrack(
                            video_bytes=video_bytes,
                            original_name=uploaded_video.name,
                            loaded_model=model,
                            model_path=active_model_path,
                            model_to_dataset_id=st.session_state.active_model_classes.get(
                                "id_to_dataset_id", {}
                            ),
                            id_to_model_name=st.session_state.active_model_classes.get(
                                "id_to_name", {}
                            ),
                            confidence=conf_threshold,
                            iou=iou_threshold,
                            frame_stride=int(video_frame_stride),
                            sampling_seconds=video_sampling_seconds,
                            max_samples=int(video_max_samples),
                            prefer_gpu=prefer_gpu_inference,
                            sam2_predictor=sam2_predictor,
                            max_sam2_tracks=int(video_max_sam2_tracks),
                            progress_callback=update_video_progress,
                        )
                    video_analysis["sam2_status"] = sam2_status
                    st.session_state.video_analysis = video_analysis
                    st.session_state.last_inference_device = video_analysis["used_device"]
                    st.session_state.last_cuda_fallback_error = video_analysis.get(
                        "cuda_fallback_error"
                    )
                    st.success("Analyse vidéo terminée.")
                except Exception as error:
                    st.error(f"Erreur pendant l'analyse vidéo : {error}")

        video_analysis = st.session_state.video_analysis
        if video_analysis:
            tracks = video_analysis.get("tracks", [])
            metric_cols = st.columns(4)
            metric_cols[0].metric("Frames vidéo", video_analysis.get("total_frames", 0))
            metric_cols[1].metric("Frames analysées", video_analysis.get("sampled_frames", 0))
            metric_cols[2].metric("Pistes suivies", len(tracks))
            metric_cols[3].metric(
                "Meilleures frames", len({track.get("frame_index") for track in tracks})
            )
            timing_cols = st.columns(4)
            timing_cols[0].metric(
                "YOLO + tracking", f"{video_analysis.get('tracking_elapsed_seconds', 0):.1f} s"
            )
            timing_cols[1].metric(
                "SAM2", f"{video_analysis.get('sam2_elapsed_seconds', 0):.1f} s"
            )
            timing_cols[2].metric(
                "Temps total", f"{video_analysis.get('total_elapsed_seconds', 0):.1f} s"
            )
            timing_cols[3].metric("Calcul", video_analysis.get("used_device", "-") )
            sam2_status = video_analysis.get("sam2_status", "SAM2 non chargé")
            if any(track.get("mask_polygon") for track in tracks):
                st.success(sam2_status)
            else:
                st.warning(sam2_status)
            segmented_count = video_analysis.get("sam2_tracks_processed", 0)
            if segmented_count < len(tracks) and segmented_count > 0:
                st.info(
                    f"SAM2 a segmenté les {segmented_count} meilleures pistes sur {len(tracks)}. "
                    "Augmentez la limite pour segmenter les autres."
                )

            ignored_predictions = video_analysis.get("ignored_predictions", {})
            if ignored_predictions:
                ignored_text = ", ".join(
                    f"{name} ({count})" for name, count in ignored_predictions.items()
                )
                st.warning(f"Classes ignorées car hors dataset actif : {ignored_text}")
            if video_analysis.get("sam2_errors"):
                with st.expander("Erreurs SAM2 sur certaines pistes"):
                    for error in video_analysis["sam2_errors"]:
                        st.write(error)

            if tracks:
                track_options = {
                    (
                        f"{track['track_id']} | {get_dataset_class_display(track['class_id'])} | "
                        f"frame {track['frame_index']} | {track['confidence']:.0%}"
                    ): index
                    for index, track in enumerate(tracks)
                }
                selected_track_label = st.selectbox(
                    "Défaut suivi / meilleure frame",
                    list(track_options),
                    key="video_selected_track",
                )
                selected_track = tracks[track_options[selected_track_label]]
                frame_array = np.frombuffer(selected_track["frame_jpeg"], np.uint8)
                selected_frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if selected_frame_bgr is not None:
                    st.image(
                        render_video_track_preview(selected_frame_bgr, selected_track),
                        caption=(
                            f"Meilleure frame à {selected_track['timestamp_seconds']:.2f}s "
                            f"sur {selected_track['frames_seen']} apparition(s)"
                        ),
                        use_container_width=True,
                    )
                    img_bgr = selected_frame_bgr
                    video_stem = os.path.splitext(
                        os.path.basename(video_analysis["video_name"])
                    )[0]
                    safe_stem = "".join(
                        char if char.isalnum() else "_" for char in video_stem
                    ).strip("_") or "video"
                    safe_track = "".join(
                        char if char.isalnum() else "_" for char in selected_track["track_id"]
                    ).strip("_")
                    filename = (
                        f"{safe_stem}_{safe_track}_frame_{selected_track['frame_index']:07d}.jpg"
                    )
                    editor_key = (
                        f"{st.session_state.video_upload_hash}:{selected_track['track_id']}:"
                        f"{selected_track['frame_index']}"
                    )
                    if st.session_state.get("video_editor_key") != editor_key:
                        st.session_state.video_editor_key = editor_key
                        st.session_state.video_editor_boxes = [
                            {
                                "class_id": selected_track["class_id"],
                                "model_class_id": selected_track["model_class_id"],
                                "model_class_name": selected_track["model_class_name"],
                                "box": [int(value) for value in selected_track["bbox"]],
                                "conf": selected_track["confidence"],
                                "track_id": selected_track["track_id"],
                                "mask_polygon": selected_track.get("mask_polygon"),
                                "mask_confidence": selected_track.get("mask_confidence"),
                            }
                        ]
                else:
                    st.error("Impossible de relire la meilleure frame sélectionnée.")
            else:
                st.info("Aucun défaut suivi n'a été détecté dans les frames analysées.")

    else:
        uploaded_batch = st.file_uploader(
            "Déposez plusieurs photos ou un fichier ZIP",
            type=["zip", "jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="uploader_batch"
        )
        batch_sam2_col, batch_sam2_limit_col = st.columns(2)
        with batch_sam2_col:
            image_enable_sam2 = st.checkbox(
                "Activer SAM2 sur le lot",
                value=True,
                key="batch_enable_sam2",
                help="Chaque image du lot garde YOLO pour les boîtes, puis SAM2 ajoute les masques.",
            )
        with batch_sam2_limit_col:
            image_max_sam2_boxes = st.number_input(
                "Boîtes maximum à segmenter par image",
                1,
                50,
                20,
                1,
                disabled=not image_enable_sam2,
                key="batch_max_sam2_boxes",
            )
        if uploaded_batch:
            is_zip = len(uploaded_batch) == 1 and uploaded_batch[0].name.endswith('.zip')
            new_batch_files = []
            if is_zip:
                try:
                    zip_file = uploaded_batch[0]
                    zip_file.seek(0)
                    import zipfile
                    import io
                    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
                        for zname in z.namelist():
                            ext = os.path.splitext(zname)[1].lower()
                            if ext in ['.jpg', '.jpeg', '.png', '.webp'] and not zname.startswith('__MACOSX') and not os.path.basename(zname).startswith('.'):
                                file_bytes = z.read(zname)
                                new_batch_files.append({
                                    "name": os.path.basename(zname),
                                    "bytes": file_bytes
                                })
                except Exception as e:
                    st.error(f"Erreur de décompression du ZIP : {e}")
            else:
                for f in uploaded_batch:
                    ext = os.path.splitext(f.name)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                        f.seek(0)
                        new_batch_files.append({
                            "name": f.name,
                            "bytes": f.read()
                        })
            
            # Update state if new batch is uploaded
            existing_names = [img["name"] for img in st.session_state.batch_images]
            uploaded_names = [img["name"] for img in new_batch_files]
            if uploaded_names and uploaded_names != existing_names:
                st.session_state.batch_images = new_batch_files
                st.session_state.batch_index = 0
                st.session_state.tab1_boxes = []
                st.session_state.tab1_current_file = None
                st.session_state.tab1_current_image_key = None
                st.session_state.canvas_key += 1
                st.rerun()
                
        if st.session_state.batch_images:
            total_imgs = len(st.session_state.batch_images)
            curr_idx = st.session_state.batch_index
            
            # Progress display
            st.markdown(f"### 📁 Lot en cours : image `{curr_idx + 1} / {total_imgs}`")
            st.progress((curr_idx + 1) / total_imgs)
            
            # Navigation buttons
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button("⬅️ Précédente", disabled=(curr_idx == 0), use_container_width=True):
                    st.session_state.batch_index -= 1
                    st.session_state.canvas_key += 1
                    st.rerun()
            with nav_col2:
                st.write("") # placeholder
            with nav_col3:
                if st.button("⏭️ Passer", disabled=(curr_idx == total_imgs - 1), use_container_width=True):
                    st.session_state.batch_index += 1
                    st.session_state.canvas_key += 1
                    st.rerun()
                    
            current_image_data = st.session_state.batch_images[curr_idx]
            filename = current_image_data["name"]
            file_bytes = np.frombuffer(current_image_data["bytes"], np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error(f"Impossible de lire l'image `{filename}`. Elle sera ignorée tant que le fichier reste invalide.")
                filename = None

    if img_bgr is not None:
        # Read Image and get original dimensions
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        orig_w, orig_h = pil_img.size
        
        filename_no_ext, _ = os.path.splitext(filename)
        labels_dir = os.path.join(base_dir, "Data7.off", "labels")
        existing_label_path = os.path.join(labels_dir, f"{filename_no_ext}.txt")
        force_flag_key = f"tab1_force_reanalyse_{filename}"
        force_analysis = st.session_state.get(force_flag_key, False)
        existing_annotations_available = os.path.exists(existing_label_path)
        image_content_hash = hashlib.sha256(img_bgr.tobytes()).hexdigest()[:16]
        current_image_key = f"{work_mode}:{filename}:{image_content_hash}"
        new_tab1_file = st.session_state.get("tab1_current_image_key") != current_image_key

        if new_tab1_file:
            st.session_state.tab1_current_file = filename
            st.session_state.tab1_current_image_key = current_image_key
            st.session_state.tab1_boxes = (
                [dict(item) for item in st.session_state.video_editor_boxes]
                if work_mode == "🎥 Une vidéo"
                else []
            )
            st.session_state.tab1_history = []  # Clear history on new image
            st.session_state.canvas_key += 1
            st.session_state.tab1_loaded_existing_annotations[filename] = False

        should_run_inference = work_mode != "🎥 Une vidéo" and (
            new_tab1_file or force_analysis
        )

        if work_mode != "🎥 Une vidéo" and existing_annotations_available and not force_analysis:
            should_run_inference = False
            if not st.session_state.tab1_boxes:
                parsed_boxes = []
                try:
                    with open(existing_label_path, "r", encoding="utf-8") as f_label:
                        for line in f_label:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            box_w = float(parts[3])
                            box_h = float(parts[4])
                            x1 = int((x_center - box_w / 2) * orig_w)
                            y1 = int((y_center - box_h / 2) * orig_h)
                            x2 = int((x_center + box_w / 2) * orig_w)
                            y2 = int((y_center + box_h / 2) * orig_h)
                            x1, x2 = max(0, min(x1, orig_w)), max(0, min(x2, orig_w))
                            y1, y2 = max(0, min(y1, orig_h)), max(0, min(y2, orig_h))
                            parsed_boxes.append({
                                'class_id': cls_id,
                                'box': [x1, y1, x2, y2],
                                'conf': 1.0
                            })
                    st.session_state.tab1_boxes = parsed_boxes
                except Exception as ex:
                    st.warning(f"Impossible de recharger les annotations sauvegardées : {ex}")
                st.session_state.tab1_loaded_existing_annotations[filename] = True

        if should_run_inference:
            results = None
            if model is None:
                st.error("Aucun modèle YOLO n'est chargé. Vérifiez que `best_2.pt` est disponible.")
            else:
                try:
                    with st.spinner("🚀 Analyse IA et détection automatique des dommages en cours..."):
                        results, used_device, cuda_fallback_error = run_inference_safely(
                            model,
                            pil_img,
                            conf_threshold,
                            iou_threshold,
                            active_model_path,
                            prefer_gpu=prefer_gpu_inference,
                        )
                        st.session_state.last_inference_device = used_device
                        st.session_state.last_cuda_fallback_error = cuda_fallback_error
                    if cuda_fallback_error:
                        st.warning(
                            "Le contexte CUDA a rencontré une erreur. "
                            "L'analyse a été terminée automatiquement sur CPU."
                        )
                except Exception as ex:
                    st.session_state.last_inference_device = "Échec"
                    st.error(f"Erreur pendant l'analyse IA : {ex}")

            model_to_dataset_id = st.session_state.active_model_classes.get("id_to_dataset_id", {})
            id_to_model_name = st.session_state.active_model_classes.get("id_to_name", {})
            ignored_predictions = defaultdict(int)
            st.session_state.tab1_boxes = []
            if results is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    model_cls_id = int(box.cls[0])
                    cls_id = model_to_dataset_id.get(model_cls_id)
                    conf_val = float(box.conf[0])
                    if cls_id is None:
                        ignored_predictions[id_to_model_name.get(model_cls_id, f"class_{model_cls_id}")] += 1
                        continue
                    st.session_state.tab1_boxes.append({
                        'class_id': cls_id,
                        'model_class_id': model_cls_id,
                        'model_class_name': id_to_model_name.get(model_cls_id, f"class_{model_cls_id}"),
                        'box': [x1, y1, x2, y2],
                        'conf': conf_val
                    })
            if ignored_predictions:
                ignored_text = ", ".join([f"{name} ({count})" for name, count in ignored_predictions.items()])
                st.warning(
                    "Certaines détections ne correspondent pas aux classes de `best_2.pt` "
                    f"et n'ont pas été ajoutées : {ignored_text}."
                )
            if (
                work_mode in {"📷 Une image", "📁 Plusieurs images ou ZIP"}
                and image_enable_sam2
                and st.session_state.tab1_boxes
            ):
                with st.spinner("🧩 Segmentation SAM2 des boîtes détectées..."):
                    sam2_predictor, sam2_status = load_video_sam2_predictor()
                    if sam2_predictor is None:
                        st.warning(sam2_status)
                    else:
                        st.session_state.tab1_boxes, sam2_summary = apply_sam2_to_boxes(
                            img_bgr,
                            st.session_state.tab1_boxes,
                            sam2_predictor,
                            max_boxes=int(image_max_sam2_boxes),
                        )
                        if sam2_summary["processed"]:
                            st.success(
                                f"{sam2_status} — {sam2_summary['processed']} masque(s) ajouté(s)."
                            )
                        if sam2_summary["errors"]:
                            st.warning(
                                "SAM2 n'a pas réussi toutes les boîtes : "
                                + " | ".join(sam2_summary["errors"][:3])
                            )
            st.session_state[force_flag_key] = False
            st.session_state.tab1_loaded_existing_annotations[filename] = False
        else:
            if force_analysis:
                st.session_state[force_flag_key] = False

        existing_warning_active = (
            work_mode != "🎥 Une vidéo" and existing_annotations_available and not force_analysis
        )

        if existing_warning_active:
            st.warning(
                "Cette image possède déjà des annotations validées. Les boîtes affichées proviennent du fichier sauvegardé. "
                "Relancez l'analyse pour générer de nouvelles prédictions avec le modèle actif si besoin."
            )
            action_col1, action_col2 = st.columns([2, 1])
            with action_col1:
                if st.button("Relancer l'analyse IA", key=f"rerun_active_model_{filename}", use_container_width=True):
                    st.session_state[force_flag_key] = True
                    st.session_state.tab1_boxes = []
                    st.session_state.tab1_history = []
                    st.session_state.canvas_key += 1
                    st.rerun()
            with action_col2:
                st.caption("Annotations actuelles conservées.")

        # Main Dashboard Metrics Header
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"**Image :** `{filename}`")
            st.markdown(f"**Dimensions :** `{orig_w}x{orig_h} px`")
        with m_col2:
            st.markdown(f"**Boîtes détectées :** <span class='metric-value'>{len(st.session_state.tab1_boxes)}</span>", unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"**Classe active :** <span style='color:{active_color}; font-weight:800;'>● {get_dataset_class_display(selected_class_id)}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Main Canvas Editor Setup
        canvas_width = 800
        scale_factor = canvas_width / orig_w
        canvas_height = int(orig_h * scale_factor)
        
        col_canvas, col_panel = st.columns([3, 2])
        
        with col_canvas:
            st.subheader("🖼️ Image et boîtes")
            
            # Undo button at the top of the canvas
            u_col1, u_col2 = st.columns([1, 4])
            with u_col1:
                undo_disabled = len(st.session_state.tab1_history) == 0
                if st.button("↩️ Annuler", key="undo_tab1", disabled=undo_disabled, use_container_width=True):
                    # Pop from history and restore
                    st.session_state.tab1_boxes = st.session_state.tab1_history.pop()
                    st.session_state.canvas_key += 1
                    st.success("Action annulée ! ↩️")
            with u_col2:
                st.markdown("<p style='color:#94A3B8; margin-top:5px;'><em>Cliquez-glissez sur l'image pour ajouter une boîte.</em></p>", unsafe_allow_html=True)
            
            # Display background with existing boxes drawn on it for preview
            preview_img = img_rgb.copy()
            for idx, item in enumerate(st.session_state.tab1_boxes):
                bx1, by1, bx2, by2 = item['box']
                cid = item['class_id']
                cf = item['conf']
                c_hex = get_dataset_class_color(cid).lstrip('#')
                c_rgb = tuple(int(c_hex[i:i+2], 16) for i in (0, 2, 4))

                mask_polygon = item.get("mask_polygon")
                if mask_polygon and len(mask_polygon) >= 3:
                    mask_points = np.asarray(mask_polygon, dtype=np.int32)
                    mask_overlay = preview_img.copy()
                    cv2.fillPoly(mask_overlay, [mask_points], c_rgb)
                    preview_img = cv2.addWeighted(mask_overlay, 0.24, preview_img, 0.76, 0)
                    cv2.polylines(preview_img, [mask_points], True, c_rgb, 3, cv2.LINE_AA)
                 
                # Draw box
                thickness = max(2, int(min(orig_h, orig_w) * 0.005))
                cv2.rectangle(preview_img, (bx1, by1), (bx2, by2), c_rgb, thickness)
                
                # Draw label
                lbl = f"[{idx+1}] {get_dataset_class_display(cid)}"
                f_scale = max(0.4, min(orig_h, orig_w) * 0.0006)
                f_thickness = max(1, int(thickness / 2))
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_thickness)
                cv2.rectangle(preview_img, (bx1, by1 - th - 10), (bx1 + tw + 10, by1), c_rgb, -1)
                cv2.putText(preview_img, lbl, (bx1 + 5, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
                
            preview_pil = Image.fromarray(preview_img)
            
            # Streamlit Drawable Canvas with dynamic key to allow reset
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.15)",  # Translucent fill
                stroke_width=3,
                stroke_color=active_color,
                background_image=preview_pil,
                update_streamlit=True,
                width=canvas_width,
                height=canvas_height,
                drawing_mode="rect",
                key=f"canvas_tab1_{st.session_state.canvas_key}",
            )
            
            # Process newly drawn shapes from canvas
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if len(objects) > 0:
                    # Get the last drawn object (new rectangle)
                    last_obj = objects[-1]
                    if last_obj["type"] == "rect":
                        image_box = canvas_rect_to_image_box(
                            last_obj, scale_factor, orig_w, orig_h
                        )
                        if image_box is None:
                            st.warning("La boîte dessinée est trop petite.")
                        else:
                            x1, y1, x2, y2 = image_box
                            # Verify if this box is already added to avoid duplicates on refresh
                            is_duplicate = False
                            for existing in st.session_state.tab1_boxes:
                                ex_x1, ex_y1, ex_x2, ex_y2 = existing['box']
                                if abs(ex_x1 - x1) < 5 and abs(ex_y1 - y1) < 5 and abs(ex_x2 - x2) < 5 and abs(ex_y2 - y2) < 5:
                                    is_duplicate = True
                                    break

                        if image_box is not None and not is_duplicate:
                            # Save history BEFORE adding
                            save_history("tab1")

                            st.session_state.tab1_boxes.append({
                                'class_id': selected_class_id,
                                'box': [x1, y1, x2, y2],
                                'conf': 1.0  # Manual boxes are 100% confident
                            })
                            # Increment canvas key to completely clear/reset canvas drawings
                            st.session_state.canvas_key += 1
                            st.success(f"BBox '{get_dataset_class_display(selected_class_id)}' ajoutée avec succès ! 🎉")
                            st.rerun()

        with col_panel:
            st.subheader("📋 Boîtes à vérifier")
            
            if len(st.session_state.tab1_boxes) == 0:
                st.info("Aucune boîte détectée. Vous pouvez en dessiner une directement sur l'image.")
            else:
                class_options = available_class_ids
                for idx, item in enumerate(st.session_state.tab1_boxes):
                    cid = item['class_id']
                    bx1, by1, bx2, by2 = item['box']
                    cf = item['conf']
                    
                    display_label = get_dataset_class_display(cid)
                    c_hex = get_dataset_class_color(cid)
                    
                    # Render box information card
                    st.markdown(f"""
                    <div class='box-card' style='border-left: 5px solid {c_hex};'>
                        <div style='display: flex; justify-content: space-between;'>
                            <strong>#{idx+1} : {display_label}</strong>
                            <span style='background-color: {c_hex}33; color: {c_hex}; border-radius: 12px; padding: 2px 8px; font-size: 0.8rem; font-weight:700;'>
                                {cf:.1%} Conf.
                            </span>
                        </div>
                        <div style='color: #94A3B8; font-size: 0.85rem; margin-top: 5px;'>
                            Position : [X: {bx1} ➜ {bx2}] | [Y: {by1} ➜ {by2}]
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Actions for this box
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        box_class_options = class_options if cid in class_options else [cid] + class_options
                        current_index = box_class_options.index(cid)
                        new_cls = st.selectbox(
                            f"Changer classe #{idx+1}",
                            box_class_options,
                            index=current_index,
                            key=f"cls_sel_{idx}",
                            format_func=get_dataset_class_display
                        )
                        if new_cls != cid:
                            # Save history before changing class
                            save_history("tab1")
                            st.session_state.tab1_boxes[idx]['class_id'] = new_cls
                    with b_col2:
                        if st.button(f"❌ Supprimer #{idx+1}", key=f"del_box_{idx}", use_container_width=True):
                            # Save history before deletion
                            save_history("tab1")
                            st.session_state.tab1_boxes.pop(idx)
                            st.warning(f"BBox #{idx+1} supprimée !")

            if work_mode in {"📷 Une image", "📁 Plusieurs images ou ZIP"}:
                st.markdown("---")
                st.markdown("### 🧩 Segmentation SAM2")
                sam2_flash = st.session_state.pop("image_sam2_flash", None)
                if sam2_flash:
                    level, message = sam2_flash
                    if level == "success":
                        st.success(message)
                    else:
                        st.warning(message)
                mask_count = sum(
                    1
                    for item in st.session_state.tab1_boxes
                    if item.get("mask_polygon")
                )
                st.caption(
                    f"{mask_count}/{len(st.session_state.tab1_boxes)} boîte(s) avec masque SAM2."
                )
                if st.button(
                    "🧩 Segmenter les boîtes avec SAM2",
                    use_container_width=True,
                    disabled=not st.session_state.tab1_boxes,
                ):
                    sam2_predictor, sam2_status = load_video_sam2_predictor()
                    if sam2_predictor is None:
                        st.session_state.image_sam2_flash = ("warning", sam2_status)
                    else:
                        with st.spinner("Segmentation SAM2 en cours..."):
                            updated_boxes, sam2_summary = apply_sam2_to_boxes(
                                img_bgr,
                                st.session_state.tab1_boxes,
                                sam2_predictor,
                                max_boxes=int(image_max_sam2_boxes),
                                only_missing=True,
                            )
                        st.session_state.tab1_boxes = updated_boxes
                        if sam2_summary["processed"]:
                            st.session_state.image_sam2_flash = (
                                "success",
                                f"{sam2_status} — {sam2_summary['processed']} masque(s) ajouté(s).",
                            )
                        else:
                            st.session_state.image_sam2_flash = (
                                "warning",
                                "Aucun nouveau masque ajouté. Les boîtes sont peut-être déjà segmentées.",
                            )
                    st.rerun()

            st.markdown("---")
            st.markdown("### 💾 Sauvegarder dans Data7.off")
            st.write("Enregistre l'image et ses boîtes au format YOLO.")
            
            # Determine button label based on work mode and batch progress
            if work_mode in {"📷 Une image", "🎥 Une vidéo"}:
                btn_label = (
                    "📦 Enregistrer la meilleure frame"
                    if work_mode == "🎥 Une vidéo"
                    else "📦 Enregistrer"
                )
            else:
                total_imgs = len(st.session_state.batch_images)
                curr_idx = st.session_state.batch_index
                if curr_idx < total_imgs - 1:
                    btn_label = "📦 Enregistrer et passer à la suivante"
                else:
                    btn_label = "🏁 Enregistrer et terminer le lot"
            
            if st.button(btn_label, type="primary", use_container_width=True):
                invalid_ids = find_nonstandard_class_ids(st.session_state.tab1_boxes)
                if invalid_ids:
                    st.error(
                        "Impossible d'enregistrer : corrigez d'abord les classes non standard "
                        + ", ".join([f"#{cid}" for cid in invalid_ids])
                        + " vers une classe de `best_2.pt`."
                    )
                else:
                    persist_info = persist_annotation(
                        filename,
                        st.session_state.tab1_boxes,
                        orig_w,
                        orig_h,
                        image_bgr=img_bgr,
                        source=("video_tracking_sam2" if work_mode == "🎥 Une vidéo" else "tab1")
                    )
                    label_filename = os.path.basename(persist_info.get("label_path", ""))
                    metadata_path = persist_info.get("metadata_path")
                    rel_metadata_path = os.path.relpath(metadata_path, base_dir) if metadata_path else ""
                    split_sync = persist_info.get("split_sync")

                    if work_mode in {"📷 Une image", "🎥 Une vidéo"}:
                        st.balloons()
                        st.success(f"🎉 Enregistré avec succès dans 'Data7.off/' !\nFichier label: `{label_filename}`")
                        if metadata_path:
                            st.caption(f"📄 Métadonnées synchronisées : `{rel_metadata_path}`")
                        if split_sync:
                            st.caption(f"🧪 Copie split mise à jour : `{split_sync['split']}`")
                    else:
                        st.toast(f"✅ Sauvegardé : {filename}")
                        if metadata_path:
                            st.toast(f"🧾 Métadonnées : {os.path.basename(metadata_path)}")
                        if curr_idx < total_imgs - 1:
                            st.session_state.batch_index += 1
                            st.session_state.canvas_key += 1
                            st.rerun()
                        else:
                            st.balloons()
                            st.success("🎉 Félicitations ! Le lot entier a été traité avec succès et injecté dans Data7.off !")
                            if metadata_path:
                                st.caption(f"📄 Dernière fiche métadonnées : `{rel_metadata_path}`")
                            # Reset batch state
                            st.session_state.batch_images = []
                            st.session_state.batch_index = 0
                            st.session_state.tab1_current_file = None
                            st.session_state.tab1_current_image_key = None
                            st.session_state.tab1_boxes = []
                            st.session_state.canvas_key += 1
                            st.rerun()

    else:
        # Premium Welcome Dashboard
        st.info("👋 Importez une photo ou un lot d'images pour commencer.")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("<div class='premium-card' style='height: 350px;'>", unsafe_allow_html=True)
            st.subheader("💡 Guide d'Annotation Interactive")
            st.markdown("""
                <div class='instruction-step'><strong>1. Chargement de l'Image</strong><br>Déposez simplement votre photo ci-dessus. L'IA se lance instantanément.</div>
                <div class='instruction-step'><strong>2. Détection automatique</strong><br>YOLOv8 propose des boîtes, puis l'app les garde dans l'ordre best_2 : crack, dent, glass shatter, lamp broken, scratch, tire flat.</div>
                <div class='instruction-step'><strong>3. Édition Tactile</strong><br>Dessinez à la souris pour corriger la pluie/reflets, ou supprimez les boîtes incorrectes.</div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_g2:
            st.markdown("<div class='premium-card' style='height: 350px;'>", unsafe_allow_html=True)
            st.subheader("📈 Format d'Exportation YOLO (Data7.off)")
            st.markdown("""
                Chaque image validée est enregistrée avec un fichier d'annotation `.txt` normalisé :
                - **Dossier Image :** `Data7.off/images/`
                - **Dossier Label :** `Data7.off/labels/`
                - **Format d'annotation :**
                  `<class_id> <x_centre> <y_centre> <largeur> <hauteur>`
                  *(Valeurs comprises entre 0.0 et 1.0)*
                - **Classes best_2 :** `0=crack`, `1=dent`, `2=glass shatter`, `3=lamp broken`, `4=scratch`, `5=tire flat`
                
                Ces fichiers sont **100% compatibles** pour relancer l'entraînement de votre modèle YOLOv8 et améliorer sa précision !
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: DATASET EXPLORER ---
if selected_section == NAV_SECTIONS[1]:
    st.subheader("📂 Dataset Data7.off")
    st.caption("Consultez les images validées, corrigez leurs boîtes, puis sauvegardez les modifications.")
    
    # Path settings
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data7_dir = os.path.join(base_dir, "Data7.off")
    images_dir = os.path.join(data7_dir, "images")
    labels_dir = os.path.join(data7_dir, "labels")
    
    # Verify if directory exists and contains images
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        st.info("ℹ️ Votre dossier 'Data7.off' est vide pour le moment. Allez sur le premier onglet pour analyser et enregistrer vos premières images.")
    else:
        # Get list of images
        saved_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Select an image to consult / edit
        selected_image_name = st.selectbox("Sélectionnez l'image à inspecter/modifier :", saved_images, key="saved_img_selector")
        
        if selected_image_name:
            # Load selected image
            selected_img_path = os.path.join(images_dir, selected_image_name)
            img_bgr_saved = cv2.imread(selected_img_path)
            if img_bgr_saved is None:
                st.error(f"Impossible de lire l'image sauvegardée : `{selected_image_name}`.")
                st.stop()
            img_rgb_saved = cv2.cvtColor(img_bgr_saved, cv2.COLOR_BGR2RGB)
            pil_img_saved = Image.fromarray(img_rgb_saved)
            saved_w, saved_h = pil_img_saved.size
            
            # Check if this is a new image selection to parse labels once
            if st.session_state.tab2_current_file != selected_image_name:
                st.session_state.tab2_current_file = selected_image_name
                st.session_state.tab2_boxes = []
                st.session_state.tab2_history = []  # Clear history on image switch
                st.session_state.canvas_key += 1
                
                # Check if corresponding YOLO label text file exists
                filename_no_ext, _ = os.path.splitext(selected_image_name)
                txt_label_path = os.path.join(labels_dir, f"{filename_no_ext}.txt")
                
                if os.path.exists(txt_label_path):
                    with open(txt_label_path, "r") as f_label:
                        lines = f_label.read().strip().splitlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                box_w = float(parts[3])
                                box_h = float(parts[4])
                                
                                # Convert normalized YOLO coordinates back to original pixel dimensions
                                x1 = int((x_center - box_w / 2) * saved_w)
                                y1 = int((y_center - box_h / 2) * saved_h)
                                x2 = int((x_center + box_w / 2) * saved_w)
                                y2 = int((y_center + box_h / 2) * saved_h)
                                
                                # Enforce image bounds
                                x1, x2 = max(0, min(x1, saved_w)), max(0, min(x2, saved_w))
                                y1, y2 = max(0, min(y1, saved_h)), max(0, min(y2, saved_h))
                                
                                st.session_state.tab2_boxes.append({
                                    'class_id': cls_id,
                                    'box': [x1, y1, x2, y2],
                                    'conf': 1.0  # Already validated labels have 100% confidence
                                })
            
            # Header Panel
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                st.markdown(f"**Fichier ouvert :** `{selected_image_name}`")
                st.markdown(f"**Dimensions :** `{saved_w}x{saved_h} px`")
            with e_col2:
                st.markdown(f"**Nombre de BBoxes enregistrées :** <span class='metric-value'>{len(st.session_state.tab2_boxes)}</span>", unsafe_allow_html=True)
            with e_col3:
                st.markdown(f"**Classe active :** <span style='color:{active_color}; font-weight:800;'>● {get_dataset_class_display(selected_class_id)}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Setup Canvas Dimensions
            canvas_width_saved = 800
            scale_factor_saved = canvas_width_saved / saved_w
            canvas_height_saved = int(saved_h * scale_factor_saved)
            
            col_canvas_saved, col_panel_saved = st.columns([3, 2])
            
            with col_canvas_saved:
                st.subheader("🖼️ Image et correction")
                
                # Undo button at the top of the canvas for Tab 2
                u2_col1, u2_col2 = st.columns([1, 4])
                with u2_col1:
                    undo2_disabled = len(st.session_state.tab2_history) == 0
                    if st.button("↩️ Annuler", key="undo_tab2", disabled=undo2_disabled, use_container_width=True):
                        st.session_state.tab2_boxes = st.session_state.tab2_history.pop()
                        st.session_state.canvas_key += 1
                        st.success("Action annulée ! ↩️")
                with u2_col2:
                    st.markdown("<p style='color:#94A3B8; margin-top:5px;'><em>Cliquez-glissez pour ajouter une boîte.</em></p>", unsafe_allow_html=True)
                
                # Render background with current boxes drawn
                preview_img_saved = img_rgb_saved.copy()
                for idx, item in enumerate(st.session_state.tab2_boxes):
                    bx1, by1, bx2, by2 = item['box']
                    cid = item['class_id']
                    cf = item['conf']
                    c_hex = get_dataset_class_color(cid).lstrip('#')
                    c_rgb = tuple(int(c_hex[i:i+2], 16) for i in (0, 2, 4))
                    
                    # Draw box
                    thickness = max(2, int(min(saved_h, saved_w) * 0.005))
                    cv2.rectangle(preview_img_saved, (bx1, by1), (bx2, by2), c_rgb, thickness)
                    
                    # Draw label
                    lbl = f"[{idx+1}] {get_dataset_class_display(cid)}"
                    f_scale = max(0.4, min(saved_h, saved_w) * 0.0006)
                    f_thickness = max(1, int(thickness / 2))
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_thickness)
                    cv2.rectangle(preview_img_saved, (bx1, by1 - th - 10), (bx1 + tw + 10, by1), c_rgb, -1)
                    cv2.putText(preview_img_saved, lbl, (bx1 + 5, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
                
                preview_pil_saved = Image.fromarray(preview_img_saved)
                
                # Streamlit Drawable Canvas for saved image editing
                canvas_result_saved = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.15)",
                    stroke_width=3,
                    stroke_color=active_color,
                    background_image=preview_pil_saved,
                    update_streamlit=True,
                    width=canvas_width_saved,
                    height=canvas_height_saved,
                    drawing_mode="rect",
                    key=f"canvas_tab2_{st.session_state.canvas_key}",
                )
                
                # Capture drawings
                if canvas_result_saved.json_data is not None:
                    objects_saved = canvas_result_saved.json_data["objects"]
                    if len(objects_saved) > 0:
                        last_obj_saved = objects_saved[-1]
                        if last_obj_saved["type"] == "rect":
                            image_box = canvas_rect_to_image_box(
                                last_obj_saved, scale_factor_saved, saved_w, saved_h
                            )
                            if image_box is None:
                                st.warning("La boîte dessinée est trop petite.")
                            else:
                                x1, y1, x2, y2 = image_box
                                is_duplicate = False
                                for existing in st.session_state.tab2_boxes:
                                    ex_x1, ex_y1, ex_x2, ex_y2 = existing['box']
                                    if abs(ex_x1 - x1) < 5 and abs(ex_y1 - y1) < 5 and abs(ex_x2 - x2) < 5 and abs(ex_y2 - y2) < 5:
                                        is_duplicate = True
                                        break

                            if image_box is not None and not is_duplicate:
                                # Save history before adding
                                save_history("tab2")

                                st.session_state.tab2_boxes.append({
                                    'class_id': selected_class_id,
                                    'box': [x1, y1, x2, y2],
                                    'conf': 1.0
                                })
                                st.session_state.canvas_key += 1
                                st.success("BBox dessinée ajoutée ! 🎉")
                                st.rerun()
                                
            with col_panel_saved:
                st.subheader("📋 Boîtes enregistrées")
                
                class_options_saved = available_class_ids
                for idx, item in enumerate(st.session_state.tab2_boxes):
                    cid = item['class_id']
                    bx1, by1, bx2, by2 = item['box']
                    cf = item['conf']
                    
                    display_label = get_dataset_class_display(cid)
                    c_hex = get_dataset_class_color(cid)
                    
                    # Box Info Card
                    st.markdown(f"""
                    <div class='box-card' style='border-left: 5px solid {c_hex};'>
                        <div style='display: flex; justify-content: space-between;'>
                            <strong>#{idx+1} : {display_label}</strong>
                            <span style='background-color: {c_hex}33; color: {c_hex}; border-radius: 12px; padding: 2px 8px; font-size: 0.8rem; font-weight:700;'>
                                Enregistrée
                            </span>
                        </div>
                        <div style='color: #94A3B8; font-size: 0.85rem; margin-top: 5px;'>
                            Position : [X: {bx1} ➜ {bx2}] | [Y: {by1} ➜ {by2}]
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Actions for this box
                    eb_col1, eb_col2 = st.columns(2)
                    with eb_col1:
                        saved_box_options = class_options_saved if cid in class_options_saved else [cid] + class_options_saved
                        current_saved_index = saved_box_options.index(cid)
                        new_cls = st.selectbox(
                            f"Changer classe #{idx+1} (saved)",
                            saved_box_options,
                            index=current_saved_index,
                            key=f"saved_cls_sel_{idx}",
                            format_func=get_dataset_class_display
                        )
                        if new_cls != cid:
                            # Save history before changing class
                            save_history("tab2")
                            st.session_state.tab2_boxes[idx]['class_id'] = new_cls
                    with eb_col2:
                        if st.button(f"❌ Supprimer #{idx+1}", key=f"saved_del_box_{idx}", use_container_width=True):
                            # Save history before deletion
                            save_history("tab2")
                            st.session_state.tab2_boxes.pop(idx)
                            st.warning(f"BBox #{idx+1} supprimée !")
                            
                st.markdown("---")
                st.markdown("### 💾 Sauvegarder les corrections")
                st.write("Met à jour le fichier d'annotations YOLO pour cette image.")
                
                if st.button("💾 Enregistrer les modifications", type="primary", use_container_width=True):
                    invalid_ids = find_nonstandard_class_ids(st.session_state.tab2_boxes)
                    if invalid_ids:
                        st.error(
                            "Impossible d'enregistrer : corrigez d'abord les classes non standard "
                            + ", ".join([f"#{cid}" for cid in invalid_ids])
                            + " vers une classe de `best_2.pt`."
                        )
                    else:
                        persist_info = persist_annotation(
                            selected_image_name,
                            st.session_state.tab2_boxes,
                            saved_w,
                            saved_h,
                            image_bgr=img_bgr_saved,
                            source="tab2"
                        )
                        label_filename_saved = os.path.basename(persist_info.get("label_path", ""))
                        metadata_path_saved = persist_info.get("metadata_path")
                        rel_metadata_saved = os.path.relpath(metadata_path_saved, base_dir) if metadata_path_saved else ""
                        split_sync_saved = persist_info.get("split_sync")

                        st.balloons()
                        st.success(f"✅ Modifications enregistrées avec succès dans `Data7.off/labels/{label_filename_saved}` !")
                        if metadata_path_saved:
                            st.caption(f"📄 Métadonnées mises à jour : `{rel_metadata_saved}`")
                        if split_sync_saved:
                            st.caption(f"🧪 Copie split mise à jour : `{split_sync_saved['split']}`")

# --- TAB 3: ACTIVE LEARNING RETRAINING ---
if selected_section == NAV_SECTIONS[2]:
    st.markdown("<h2 style='color:#38BDF8;'>🔄 Réentraîner le modèle</h2>", unsafe_allow_html=True)
    st.markdown("""
        Utilisez les annotations validées dans `Data7.off` pour lancer un nouvel entraînement YOLO.
        L'application prépare un vrai découpage `train / val / test` depuis ce dossier avant d'entraîner.
    """)
    st.info(
        "`Data7.off` reste le dataset corrigé principal. Avant l'entraînement, l'app copie les images dans "
        "`dataset_annote/train`, `dataset_annote/val` et `dataset_annote/test`. Le dossier `test` n'est jamais "
        "donné à YOLO pendant l'entraînement : il sert uniquement à mesurer la vraie performance."
    )
    
    gpu_info = get_gpu_environment_info()
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("GPU et environnement d'entrainement")
    gpu_col1, gpu_col2, gpu_col3, gpu_col4 = st.columns(4)
    gpu_col1.metric("Carte NVIDIA", gpu_info.get("physical_gpu") or "Non detectee")
    gpu_col2.metric(
        "VRAM",
        f"{gpu_info['memory_total_gb']:.1f} Go" if gpu_info.get("memory_total_gb") else "Inconnue",
    )
    gpu_col3.metric("PyTorch", gpu_info.get("torch_version") or "Indisponible")
    gpu_col4.metric("CUDA PyTorch", gpu_info.get("torch_cuda") or "CPU seulement")
    if gpu_info.get("cuda_available"):
        st.success(
            f"GPU pret : {gpu_info.get('device_name')} | pilote {gpu_info.get('driver')} | "
            f"CUDA {gpu_info.get('torch_cuda')}"
        )
    elif gpu_info.get("physical_gpu"):
        st.warning(
            "La carte NVIDIA est visible, mais ce Python utilise une version CPU de PyTorch. "
            "Lancez l'application avec `.venv-gpu\\Scripts\\python.exe`."
        )
    else:
        st.error("Aucun GPU NVIDIA utilisable n'est detecte.")
    st.markdown("</div>", unsafe_allow_html=True)

    saved_training_status = load_training_status()
    if saved_training_status:
        current_epoch = int(saved_training_status.get("current_epoch", 0) or 0)
        total_epochs = int(saved_training_status.get("total_epochs", 0) or 0)
        state_labels = {
            "starting": "Demarrage",
            "running": "En cours",
            "completed": "Termine",
            "interrupted": "Interrompu",
            "failed": "Echec",
        }
        state_value = saved_training_status.get("state", "inconnu")
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.subheader("Suivi du dernier entrainement")
        epoch_col, state_col, update_col = st.columns(3)
        epoch_col.metric("Epoch", f"{current_epoch} / {total_epochs or '?'}")
        state_col.metric("Etat", state_labels.get(state_value, state_value))
        update_col.metric(
            "Derniere mise a jour", saved_training_status.get("updated_at", "Inconnue")
        )
        st.progress(
            min(1.0, float(saved_training_status.get("progress", 0.0) or 0.0)),
            text=f"Progression : epoch {current_epoch} sur {total_epochs or '?'}",
        )
        if saved_training_status.get("losses"):
            st.caption("Losses de la derniere epoch enregistree")
            st.json(saved_training_status["losses"], expanded=False)
        if saved_training_status.get("metrics"):
            st.caption("Metriques de validation disponibles")
            st.json(saved_training_status["metrics"], expanded=False)
        st.markdown("</div>", unsafe_allow_html=True)

    # Helper to calculate stats
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data7_dir = os.path.join(base_dir, "Data7.off")
    images_dir = os.path.join(data7_dir, "images")
    labels_dir = os.path.join(data7_dir, "labels")
    
    img_count = 0
    class_stats = defaultdict(int)
    
    if os.path.exists(labels_dir):
        txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        img_count = len(txt_files)
        for f in txt_files:
            try:
                with open(os.path.join(labels_dir, f), "r") as file_handle:
                    for line in file_handle:
                        parts = line.strip().split()
                        if len(parts) > 0:
                            try:
                                cls_id = int(parts[0])
                                class_stats[cls_id] += 1
                            except Exception:
                                continue
            except Exception:
                pass
    invalid_training_ids = sorted([cid for cid in class_stats.keys() if cid not in FINAL_DATASET_CLASSES])
    dataset_annote_dir = os.path.join(base_dir, "dataset_annote")
    try:
        current_split_summary = summarize_dataset_split(dataset_annote_dir)
    except Exception:
        current_split_summary = {}
                
    # UI Metrics & Progress
    col_st1, col_st2 = st.columns([1, 1])
    
    with col_st1:
        st.markdown("<div class='premium-card' style='height: 230px;'>", unsafe_allow_html=True)
        st.subheader("🎯 Volume d'annotations")
        
        st.metric("Photos prêtes", img_count)
        st.caption("Comptées depuis `Data7.off/labels`. Chaque label doit avoir son image correspondante.")
        
        if img_count == 0:
            st.warning("Aucune image annotée n'est disponible pour l'entraînement.")
        elif invalid_training_ids:
            st.error(
                "Des anciennes classes non standard sont présentes : "
                + ", ".join([f"#{cid}" for cid in invalid_training_ids])
                + ". Corrigez ou harmonisez `Data7.off` avant d'entraîner."
            )
        else:
            st.success(f"Le prochain entraînement utilisera les **{img_count} images** disponibles.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_st2:
        st.markdown("<div class='premium-card' style='height: 230px;'>", unsafe_allow_html=True)
        st.subheader("📊 Répartition des classes")
        
        # Display small class breakdown chart
        if class_stats:
            chart_data = {get_dataset_class_display(cid): class_stats[cid] for cid in sorted(class_stats.keys())}
        else:
            fallback_ids = sorted(FINAL_DATASET_CLASSES.keys())
            chart_data = {get_dataset_class_display(cid): 0 for cid in fallback_ids}
        st.bar_chart(chart_data, horizontal=True, height=130)
        st.markdown("</div>", unsafe_allow_html=True)

    split_rows = []
    for split_key, split_label in [("train", "Train"), ("val", "Validation"), ("test", "Test indépendant")]:
        split_data = current_split_summary.get(split_key, {})
        split_rows.append({
            "Split": split_label,
            "Images": split_data.get("images", 0),
            "Labels": split_data.get("labels", 0),
            "Boîtes": split_data.get("boxes", 0),
        })
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("🧪 Découpage anti-fuite")
    st.dataframe(split_rows, use_container_width=True, hide_index=True)
    if current_split_summary.get("duplicate_hashes_across_splits"):
        st.error("Fuite potentielle : des images identiques existent dans plusieurs splits. Repréparez le split avant d'évaluer.")
    else:
        st.caption("Les doublons exacts et variantes Roboflow d'une même image sont gardés dans un seul split.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Controls Section
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Paramètres d'entraînement")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        retrain_target = st.selectbox(
            "Modèle de départ :",
            ["Référence unique (best_2.pt)"],
            help="Le réentraînement part toujours de best_2.pt pour garder le même ordre de classes."
        )
    with col_c2:
        epochs_input = st.number_input("Nombre d'époques (itérations)", min_value=1, max_value=300, value=20, step=5, help="Plus d'époques améliorent la précision mais prennent plus de temps.")
    with col_c3:
        cuda_available = cuda_is_available()
        device_options = ["CPU"]
        if cuda_available:
            device_options.append("GPU (CUDA)")
        device_input = st.selectbox("Processeur d'entraînement", device_options, help="Le GPU est proposé seulement si PyTorch détecte une carte Nvidia CUDA compatible.")
        if not cuda_available:
            st.caption("GPU CUDA non détecté : le réentraînement se lancera sur CPU.")
        st.checkbox("Utiliser best_2.pt comme base", value=True, disabled=True, help="Verrouillé pour garder le même ordre de classes partout.")

    split_col1, split_col2, split_col3 = st.columns(3)
    with split_col1:
        test_pct = st.slider("Part test jamais vue (%)", min_value=10, max_value=30, value=15, step=5)
    with split_col2:
        val_pct = st.slider("Part validation (%)", min_value=10, max_value=30, value=15, step=5)
    with split_col3:
        split_seed = st.number_input("Seed split", min_value=1, max_value=9999, value=42, step=1)
        lock_existing_test = st.checkbox("Garder le test déjà créé", value=True, help="Les images déjà en test restent en test et ne repartent pas dans train/val.")

    if test_pct + val_pct >= 60:
        st.warning("Gardez assez d'images pour l'entraînement : test + validation devrait rester sous 60%.")

    if st.button("📦 Préparer / actualiser le split train-val-test", use_container_width=True):
        try:
            split_result = prepare_train_val_test_split(
                base_dir,
                val_ratio=val_pct / 100,
                test_ratio=test_pct / 100,
                seed=int(split_seed),
                lock_existing_test=lock_existing_test,
                class_names=FINAL_DATASET_CLASSES,
            )
            st.success("Split prêt : `dataset_annote/train`, `val` et `test` ont été mis à jour.")
            st.json(split_result["summary"])
            st.rerun()
        except Exception as split_ex:
            st.error(f"Impossible de préparer le split : {split_ex}")
        
    # Trigger Retraining Button
    st.markdown("---")
    
    # Enable button but show warnings if photos are low
    if img_count == 0:
        st.error("❌ Vous ne pouvez pas lancer l'entraînement car le dossier 'Data7.off' ne contient aucune photo annotée.")
        st.button("🔥 Lancer le réentraînement de l'IA", disabled=True, use_container_width=True)
    elif invalid_training_ids:
        st.error(
            "❌ Réentraînement bloqué : `Data7.off` contient des classes non standard "
            + ", ".join([f"#{cid}" for cid in invalid_training_ids])
            + ". Utilisez l'onglet Dataset pour corriger ces boîtes ou l'onglet Harmoniser pour générer un dataset propre."
        )
        st.button("🔥 Lancer le réentraînement de l'IA", disabled=True, use_container_width=True)
    else:
        if img_count < 10:
            st.warning("⚠️ *Attention : Vous avez moins de 10 photos. L'entraînement fonctionnera techniquement mais le modèle risque de sur-apprendre (overfitting).*")
            
        retrain_clicked = st.button("🔥 Lancer le réentraînement de l'IA", type="primary", use_container_width=True)
        
        if retrain_clicked:
            try:
                # Build training queue
                training_queue = [{
                    "name": "Référence best_2",
                    "filename": "best_2.pt",
                    "temp_project": "runs_temp_best_2",
                    "output_prefix": "data7_from_best_2"
                }]

                live_progress = st.progress(
                    0.0, text=f"Preparation de l'entrainement : 0 / {int(epochs_input)} epochs"
                )
                live_epoch_text = st.empty()
                live_metrics = st.empty()
                save_training_status({
                    "state": "starting",
                    "model": "best_2.pt",
                    "current_epoch": 0,
                    "total_epochs": int(epochs_input),
                    "progress": 0.0,
                    "metrics": {},
                    "losses": {},
                    "gpu": get_gpu_environment_info(),
                })
                
                with st.status("🛠️ Lancement du pipeline d'Active Learning MLOps...", expanded=True) as status:
                    # 1. Create a leakage-safe train/val/test split from the corrected master dataset.
                    status.update(label="📁 Préparation du split train / val / test...", state="running")
                    split_result = prepare_train_val_test_split(
                        base_dir,
                        val_ratio=val_pct / 100,
                        test_ratio=test_pct / 100,
                        seed=int(split_seed),
                        lock_existing_test=lock_existing_test,
                        class_names=FINAL_DATASET_CLASSES,
                    )
                    yaml_path = split_result["yaml_path"]
                    split_summary = split_result["summary"]
                    train_n = split_summary.get("train", {}).get("images", 0)
                    val_n = split_summary.get("val", {}).get("images", 0)
                    test_n = split_summary.get("test", {}).get("images", 0)

                    if train_n == 0 or val_n == 0 or test_n == 0:
                        raise ValueError("Le split train/val/test est incomplet. Ajoutez plus d'images annotées avant l'entraînement.")
                    if split_summary.get("duplicate_hashes_across_splits"):
                        raise ValueError("Fuite détectée : une image identique existe dans plusieurs splits.")

                    status.write(
                        f"✅ Split prêt : `{train_n}` train, `{val_n}` validation, `{test_n}` test. "
                        "`test` est exclu de l'entraînement."
                    )
                    
                    # Determine training device
                    if device_input == "GPU (CUDA)" and cuda_is_available():
                        device_arg = 0
                        status.write("⚡ Entraînement sur GPU CUDA.")
                    else:
                        device_arg = "cpu"
                        status.write("🧠 Entraînement sur CPU.")
                    
                    import shutil
                    import time
                    
                    # 2. Sequential training of models in queue
                    for task in training_queue:
                        status.update(label=f"🔥 Création d'une version Data7 depuis {task['name']} ({task['filename']})...", state="running")
                        
                        start_weights = task['filename']
                        status.write(f"🧠 {task['name']} : Point de départ = `{start_weights}`")
                        
                        # Load model
                        patch_ultralytics_cache_pool_for_windows()
                        retrain_model = YOLO(os.path.join(base_dir, start_weights) if os.path.exists(os.path.join(base_dir, start_weights)) else start_weights)
                        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        train_project_dir = os.path.join(base_dir, task['temp_project'])
                        train_run_name = f"weights_{run_timestamp}"

                        def update_live_training(trainer):
                            payload = trainer_status_payload(
                                trainer, "running", task["filename"]
                            )
                            save_training_status(payload)
                            epoch = payload["current_epoch"]
                            total = payload["total_epochs"] or int(epochs_input)
                            live_progress.progress(
                                min(1.0, payload["progress"]),
                                text=f"Entrainement GPU : epoch {epoch} / {total}",
                            )
                            live_epoch_text.markdown(
                                f"**Epoch en cours : `{epoch} / {total}`**  "
                                f"| GPU : `{payload['gpu'].get('device_name') or device_arg}`"
                            )
                            live_metrics.json(
                                {
                                    "losses": payload.get("losses", {}),
                                    "metrics": payload.get("metrics", {}),
                                    "mise_a_jour": payload.get("updated_at"),
                                },
                                expanded=False,
                            )

                        retrain_model.add_callback("on_train_epoch_end", update_live_training)
                        
                        # Execute training
                        results = retrain_model.train(
                            data=yaml_path,
                            epochs=epochs_input,
                            imgsz=640,
                            batch=2,
                            device=device_arg,
                            workers=0,
                            verbose=True,
                            project=train_project_dir,
                            name=train_run_name,
                            exist_ok=False
                        )
                        completed_status = load_training_status()
                        completed_status.update({
                            "state": "completed",
                            "model": task["filename"],
                            "current_epoch": int(epochs_input),
                            "total_epochs": int(epochs_input),
                            "progress": 1.0,
                            "gpu": get_gpu_environment_info(),
                        })
                        save_training_status(completed_status)
                        live_progress.progress(
                            1.0,
                            text=f"Entrainement termine : {int(epochs_input)} / {int(epochs_input)} epochs",
                        )
                        live_epoch_text.markdown(
                            f"**Entrainement termine : `{int(epochs_input)} / {int(epochs_input)}` epochs.**"
                        )
                        
                        # Save as a new Data7 version instead of overwriting the source model.
                        status.update(label=f"📦 Sauvegarde de la nouvelle version Data7 pour {task['name']}...", state="running")
                        save_dir = str(getattr(getattr(retrain_model, "trainer", None), "save_dir", ""))
                        candidate_weight_paths = [
                            os.path.join(save_dir, "weights", "best.pt"),
                            os.path.join(save_dir, "best.pt"),
                            os.path.join(train_project_dir, train_run_name, "weights", "best.pt"),
                        ]
                        trained_weights_path = next((p for p in candidate_weight_paths if p and os.path.exists(p)), "")
                        
                        if os.path.exists(trained_weights_path):
                            output_filename = f"{task['output_prefix']}_{run_timestamp}.pt"
                            output_model_path = os.path.join(base_dir, output_filename)
                            shutil.copy(trained_weights_path, output_model_path)

                            expected_names = {cid: name for cid, name in FINAL_DATASET_CLASSES.items()}
                            try:
                                trained_model_check = YOLO(output_model_path)
                                trained_names = {int(k): str(v) for k, v in trained_model_check.names.items()}
                            except Exception:
                                trained_names = {}

                            if trained_names == expected_names:
                                status.write(f"✅ Modèle best_2 sauvegardé : `{output_filename}` avec l'ordre de classes de `best_2.pt`.")
                            else:
                                status.write(f"⚠️ Modèle sauvegardé : `{output_filename}`, mais ses classes déclarées doivent être vérifiées : `{trained_names}`")
                            
                            # Copy results.csv before cleaning up
                            csv_src = os.path.join(save_dir, "results.csv") if save_dir else ""
                            if os.path.exists(csv_src):
                                csv_dst = os.path.join(base_dir, f"results_{output_filename.replace('.pt', '')}.csv")
                                shutil.copy(csv_src, csv_dst)
                                status.write(f"📈 Courbes d'entraînement sauvegardées dans `{os.path.basename(csv_dst)}` !")
                            
                            # Clean up temporary training folder
                            try:
                                shutil.rmtree(train_project_dir)
                            except Exception:
                                pass
                        else:
                            status.write(f"❌ Erreur : Impossible de localiser les nouveaux poids pour {task['name']}.")
                    
                    # Clear cache and trigger reload
                    st.cache_resource.clear()
                    status.update(label="🎉 Entraînement terminé avec succès !", state="complete")
                    
                    st.balloons()
                    st.success("🏆 Entraînement terminé. Les nouvelles versions Data7 sont enregistrées séparément : évalue-les dans l'onglet Performances avant de remplacer un modèle principal.")
                    st.rerun()
                        
            except Exception as ex:
                import traceback
                failed_status = load_training_status()
                failed_status.update({
                    "state": "failed",
                    "error": str(ex),
                    "gpu": get_gpu_environment_info(),
                })
                save_training_status(failed_status)
                error_trace_path = os.path.join(base_dir, "last_training_error.txt")
                with open(error_trace_path, "w", encoding="utf-8") as ferr:
                    ferr.write(traceback.format_exc())
                st.error(f"❌ Une erreur est survenue pendant le réentraînement : {ex}")
                st.caption(f"Détail technique enregistré dans `{os.path.basename(error_trace_path)}`.")
                st.info("Astuce : Si vous avez choisi GPU (CUDA) mais qu'il y a une erreur, assurez-vous d'avoir installé les pilotes NVIDIA CUDA et PyTorch avec support CUDA, sinon réessayez en choisissant 'CPU'.")
                
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 4: MODEL PERFORMANCE ANALYSIS ---
if selected_section == NAV_SECTIONS[3]:
    st.markdown("<h2 style='color:#38BDF8;'>📊 Performances du modèle</h2>", unsafe_allow_html=True)
    st.markdown("""
        Évaluez les modèles sur un jeu de **test indépendant**, puis comparez les versions entre elles.
        Si le dossier `dataset_annote/test` est vide, l'évaluation restera bloquée.
    """)
    st.info(
        "Workflow conseillé : évalue d'abord `best_2.pt`, puis évalue le nouveau modèle `data7_from_best_2_*.pt`. "
        "Le comparateur permet ensuite de voir l'écart avant/après correction."
    )
    
    import json
    import hashlib
    import time
    import shutil
    from datetime import datetime
    import pandas as pd
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset_annote")
    history_file_path = os.path.join(base_dir, "models_history", "models_history.json")
    eval_runs_base = os.path.join(base_dir, "evaluation_runs")
    
    # Helper to calculate md5 hash
    def calculate_md5(file_path):
        try:
            h = hashlib.md5()
            with open(file_path, 'rb') as fh:
                for chunk in iter(lambda: fh.read(4096), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    # Helper function to analyze the dataset structure
    def analyze_dataset_structure():
        splits = ["train", "val", "test"]
        stats = {}
        all_image_hashes = {}
        duplicate_names = {}
        
        for split in splits:
            img_dir = os.path.join(dataset_dir, split, "images")
            lbl_dir = os.path.join(dataset_dir, split, "labels")
            
            images = []
            if os.path.exists(img_dir):
                images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                
            labels = []
            if os.path.exists(lbl_dir):
                labels = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
                
            split_hashes = {}
            missing_lbl = 0
            duplicate_fn = 0
            img_names_seen = set()
            
            class_counts = defaultdict(int)
            boxes_total = 0
            background_images_count = 0
            
            for img_name in images:
                if img_name in img_names_seen:
                    duplicate_fn += 1
                img_names_seen.add(img_name)
                
                img_path = os.path.join(img_dir, img_name)
                file_hash = calculate_md5(img_path)
                if file_hash:
                    split_hashes[img_name] = file_hash
                    
                    # Check cross-split duplicates
                    if file_hash in all_image_hashes:
                        other_split, other_name = all_image_hashes[file_hash]
                        if other_split != split:
                            if split not in duplicate_names:
                                duplicate_names[split] = []
                            duplicate_names[split].append({
                                "name": img_name,
                                "other_split": other_split,
                                "other_name": other_name
                            })
                    else:
                        all_image_hashes[file_hash] = (split, img_name)
                
                # Match label file
                img_no_ext, _ = os.path.splitext(img_name)
                lbl_file = f"{img_no_ext}.txt"
                lbl_path = os.path.join(lbl_dir, lbl_file)
                
                if not os.path.exists(lbl_path):
                    missing_lbl += 1
                else:
                    try:
                        with open(lbl_path, "r") as lf:
                            lines = lf.read().strip().splitlines()
                            if len(lines) == 0:
                                background_images_count += 1
                            else:
                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        try:
                                            cls_id = int(parts[0])
                                            class_counts[cls_id] += 1
                                            boxes_total += 1
                                        except Exception:
                                            continue
                    except Exception:
                        pass
                        
            # Check orphaned labels
            missing_img = 0
            for lbl_name in labels:
                lbl_no_ext, _ = os.path.splitext(lbl_name)
                found = False
                for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                    if os.path.exists(os.path.join(img_dir, f"{lbl_no_ext}{ext}")):
                        found = True
                        break
                if not found:
                    missing_img += 1
                    
            stats[split] = {
                "images_count": len(images),
                "labels_count": len(labels),
                "boxes_count": boxes_total,
                "background_images": background_images_count,
                "class_counts": dict(class_counts),
                "missing_labels": missing_lbl,
                "missing_images": missing_img,
                "duplicate_filenames": duplicate_fn,
                "hashes": split_hashes
            }
            
        return stats, duplicate_names

    # Run actual YOLO evaluation using Ultralytics
    def run_model_evaluation(model_file_path, split="test"):
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        split_images_dir = os.path.join(dataset_dir, split, "images")
        total_eval_images = 0
        if os.path.isdir(split_images_dir):
            total_eval_images = len([
                filename for filename in os.listdir(split_images_dir)
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ])
        
        # Instantiate model
        patch_ultralytics_cache_pool_for_windows()
        temp_model = YOLO(model_file_path)

        eval_progress = st.progress(
            0.0,
            text=f"Evaluation : 0% - 0 / {total_eval_images or '?'} images",
        )
        eval_timing = st.empty()
        eval_device = "GPU CUDA" if cuda_is_available() else "CPU"
        eval_start_time = time.time()

        def format_eval_duration(seconds):
            seconds = max(0, int(seconds))
            hours, remainder = divmod(seconds, 3600)
            minutes, secs = divmod(remainder, 60)
            if hours:
                return f"{hours}h {minutes:02d}m {secs:02d}s"
            if minutes:
                return f"{minutes}m {secs:02d}s"
            return f"{secs}s"

        def update_evaluation_progress(validator):
            try:
                batch_index = int(getattr(validator, "batch_i", 0)) + 1
                total_batches = max(1, len(getattr(validator, "dataloader", [])))
                progress_value = min(1.0, batch_index / total_batches)
                elapsed = time.time() - eval_start_time
                eta = (elapsed / progress_value) - elapsed if progress_value > 0 else 0
                processed = min(
                    total_eval_images,
                    max(1, int(round(progress_value * total_eval_images))),
                ) if total_eval_images else batch_index
                percent = int(round(progress_value * 100))
                eval_progress.progress(
                    progress_value,
                    text=(
                        f"Evaluation : {percent}% - {processed} / "
                        f"{total_eval_images or '?'} images"
                    ),
                )
                eval_timing.info(
                    f"{eval_device} | Ecoule : {format_eval_duration(elapsed)} | "
                    f"Temps restant estime : {format_eval_duration(eta)}"
                )
            except Exception:
                pass

        temp_model.add_callback("on_val_batch_end", update_evaluation_progress)
        
        project_dir = os.path.join(base_dir, "temp_eval_runs")
        run_name = "eval"
        
        with st.spinner(f"⚡ Évaluation scientifique du modèle `{os.path.basename(model_file_path)}` sur le jeu `{split}`..."):
            metrics = temp_model.val(
                data=yaml_path,
                split=split,
                imgsz=640,
                device=0 if cuda_is_available() else "cpu",
                workers=0,
                plots=True,
                save_json=True,
                project=project_dir,
                name=run_name,
                exist_ok=True
            )
        eval_elapsed = time.time() - eval_start_time
        eval_progress.progress(
            1.0,
            text=f"Evaluation terminee : 100% - {total_eval_images} / {total_eval_images} images",
        )
        eval_timing.success(
            f"{eval_device} | Duree totale : {format_eval_duration(eval_elapsed)}"
        )
        
        # Get raw results
        results_dict = metrics.results_dict
        p_global = float(results_dict.get('metrics/precision(B)', 0.0))
        r_global = float(results_dict.get('metrics/recall(B)', 0.0))
        map50_global = float(results_dict.get('metrics/mAP50(B)', 0.0))
        map50_95_global = float(results_dict.get('metrics/mAP50-95(B)', 0.0))
        f1_global = 2 * (p_global * r_global) / (p_global + r_global) if (p_global + r_global) > 0 else 0.0
        
        # Speed
        preprocess_speed = metrics.speed.get('preprocess', 0.0)
        inference_speed = metrics.speed.get('inference', 0.0)
        postprocess_speed = metrics.speed.get('postprocess', 0.0)
        avg_speed = preprocess_speed + inference_speed + postprocess_speed
        
        # Class-level metrics
        per_class_metrics = {}
        ordered_ids_eval = st.session_state.active_model_classes.get("ordered_ids", [])
        id_to_name_eval = st.session_state.active_model_classes.get("id_to_name", {})
        id_to_display_eval = st.session_state.active_model_classes.get("id_to_display", {})

        for idx, cid in enumerate(ordered_ids_eval):
            name = id_to_name_eval.get(cid, f"class_{cid}")
            display_name = id_to_display_eval.get(cid, humanize_class_name(name))

            p_cls = float(metrics.box.p[idx]) if hasattr(metrics.box, 'p') and len(getattr(metrics.box, 'p', [])) > idx else 0.0
            r_cls = float(metrics.box.r[idx]) if hasattr(metrics.box, 'r') and len(getattr(metrics.box, 'r', [])) > idx else 0.0
            map50_cls = float(metrics.box.ap50[idx]) if hasattr(metrics.box, 'ap50') and len(getattr(metrics.box, 'ap50', [])) > idx else 0.0
            map50_95_cls = float(metrics.box.ap[idx]) if hasattr(metrics.box, 'ap') and len(getattr(metrics.box, 'ap', [])) > idx else 0.0
            f1_cls = 2 * (p_cls * r_cls) / (p_cls + r_cls) if (p_cls + r_cls) > 0 else 0.0

            per_class_metrics[str(cid)] = {
                "class_id": cid,
                "class_name": name,
                "class_display": display_name,
                "precision": p_cls,
                "recall": r_cls,
                "f1": f1_cls,
                "map50": map50_cls,
                "map50_95": map50_95_cls
            }
            
        stats, _ = analyze_dataset_structure()
        test_img_count = stats.get(split, {}).get("images_count", 0)
        
        # Create output folders
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir_name = f"evaluation_{timestamp}"
        eval_dir_path = os.path.join(eval_runs_base, eval_dir_name)
        os.makedirs(eval_dir_path, exist_ok=True)
        
        # Copy plots and cleanup
        plots_copied = []
        temp_eval_dir = os.path.join(project_dir, run_name)
        if os.path.exists(temp_eval_dir):
            for f in os.listdir(temp_eval_dir):
                if f.lower().endswith(('.png', '.jpg')):
                    shutil.copy(os.path.join(temp_eval_dir, f), os.path.join(eval_dir_path, f))
                    plots_copied.append(f)
            try:
                shutil.rmtree(project_dir)
            except Exception:
                pass
                
        # Save metrics.json
        metrics_data = {
            "precision": p_global,
            "recall": r_global,
            "f1": f1_global,
            "map50": map50_global,
            "map50_95": map50_95_global,
            "inference_speed_ms": avg_speed,
            "per_class": per_class_metrics
        }
        with open(os.path.join(eval_dir_path, "metrics.json"), "w", encoding="utf-8") as fm:
            json.dump(metrics_data, fm, indent=2, ensure_ascii=False)
            
        # Save metadata.json
        metadata = {
            "model_name": os.path.basename(model_file_path),
            "model_path": os.path.abspath(model_file_path),
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": "dataset_annote/data.yaml",
            "split": split,
            "test_images_count": test_img_count,
            "imgsz": 640,
            "epochs_used_for_training": st.session_state.get("last_trained_epochs", 20)
        }
        with open(os.path.join(eval_dir_path, "metadata.json"), "w", encoding="utf-8") as fmet:
            json.dump(metadata, fmet, indent=2, ensure_ascii=False)
            
        # Append to json history
        history_data = {"history": []}
        if os.path.exists(history_file_path):
            try:
                with open(history_file_path, "r", encoding="utf-8") as fh:
                    history_data = json.load(fh)
            except Exception:
                pass
                
        new_entry = {
            "id": eval_dir_name,
            "model_name": metadata["model_name"],
            "model_path": metadata["model_path"],
            "evaluation_date": metadata["evaluation_date"],
            "test_images_count": test_img_count,
            "metrics": metrics_data,
            "metadata": metadata,
            "plots": plots_copied
        }
        history_data["history"].append(new_entry)
        
        os.makedirs(os.path.dirname(history_file_path), exist_ok=True)
        with open(history_file_path, "w", encoding="utf-8") as f_out:
            json.dump(history_data, f_out, indent=2, ensure_ascii=False)
            
        return new_entry

    # Analyze dataset splits and display info
    try:
        dataset_stats, cross_split_duplicates = analyze_dataset_structure()
    except Exception as e:
        dataset_stats, cross_split_duplicates = {}, {}
        st.error(f"Erreur d'analyse du dataset : {e}")

    # Check validation blocks
    test_stats = dataset_stats.get("test", {})
    test_img_count = test_stats.get("images_count", 0)
    test_boxes_count = test_stats.get("boxes_count", 0)
    test_bg_count = test_stats.get("background_images", 0)
    test_class_counts = dict(test_stats.get("class_counts", {}))
    
    # ------------------ SECTIONS ------------------
    col_lhs, col_rhs = st.columns([3, 2])
    
    with col_lhs:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.subheader("🤖 Modèles disponibles")
        
        # List only the best_2 reference line and models trained from it.
        root_models = [
            f for f in os.listdir(base_dir)
            if f.endswith('.pt') and (f == 'best_2.pt' or f.startswith('data7_from_best_2'))
        ]
        
        model_info_list = []
        for rm in root_models:
            path_rm = os.path.join(base_dir, rm)
            mtime_rm = os.path.getmtime(path_rm)
            dt_rm = datetime.fromtimestamp(mtime_rm).strftime("%Y-%m-%d %H:%M:%S")
            sz_rm = os.path.getsize(path_rm) / (1024 * 1024) # MB
            
            # epochs and imgsz can be found in history or defaults
            epochs_approx = "Inconnu"
            imgsz_approx = 640
            
            # Read from history if evaluated
            if os.path.exists(history_file_path):
                try:
                    with open(history_file_path, "r", encoding="utf-8") as hf:
                        h_data = json.load(hf)
                        for item in h_data.get("history", []):
                            if item["model_name"] == rm:
                                epochs_approx = item["metadata"].get("epochs_used_for_training", 20)
                                imgsz_approx = item["metadata"].get("imgsz", 640)
                                break
                except Exception:
                    pass
            
            model_info_list.append({
                "Nom": rm,
                "Taille (Mo)": f"{sz_rm:.1f} Mo",
                "Date modification": dt_rm,
                "Epochs": epochs_approx,
                "Img Size": imgsz_approx,
                "Chemin": path_rm
            })
            
        df_models = pd.DataFrame(model_info_list)
        st.dataframe(df_models, use_container_width=True, hide_index=True)
        
        # Let user upload a custom .pt file to evaluate
        st.markdown("---")
        st.write("📥 **Ajouter un autre modèle `.pt` à évaluer :**")
        uploaded_pt = st.file_uploader("Sélectionnez un modèle YOLO réentraîné", type=["pt"], key="eval_pt_uploader")
        
        if uploaded_pt:
            uploaded_pt_path = os.path.join(base_dir, uploaded_pt.name)
            if not os.path.exists(uploaded_pt_path):
                with open(uploaded_pt_path, "wb") as f_pt:
                    f_pt.write(uploaded_pt.getbuffer())
                st.success(f"Modèle importé avec succès sous `{uploaded_pt.name}` !")
                st.rerun()

        # Action Buttons
        st.markdown("---")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        # Disable buttons if test set has 0 images
        is_test_empty = test_img_count == 0
        
        with btn_col1:
            if st.button("🔍 Évaluer le modèle actif", type="secondary", disabled=is_test_empty, use_container_width=True, help="Évalue le modèle sélectionné dans le panneau latéral sur le jeu test."):
                run_path = os.path.join(base_dir, active_model_path)
                if os.path.exists(run_path):
                    with st.spinner("Lancement de la validation YOLO..."):
                        run_model_evaluation(run_path, split="test")
                    st.balloons()
                    st.success("🏆 Évaluation du modèle actuel enregistrée !")
                    st.rerun()
                else:
                    st.error(f"Fichier modèle introuvable à l'emplacement : {active_model_path}")
                    
        with btn_col2:
            other_opts = [m["Nom"] for m in model_info_list if m["Nom"] != os.path.basename(active_model_path)]
            selected_other_pt = st.selectbox("Autre modèle :", other_opts if other_opts else ["Aucun"], key="select_other_pt_eval")
            
            if st.button("🔥 Évaluer ce modèle", type="secondary", disabled=(is_test_empty or selected_other_pt == "Aucun"), use_container_width=True):
                target_path = os.path.join(base_dir, selected_other_pt)
                if os.path.exists(target_path):
                    with st.spinner("Lancement de la validation YOLO..."):
                        run_model_evaluation(target_path, split="test")
                    st.balloons()
                    st.success(f"🏆 Évaluation de {selected_other_pt} enregistrée !")
                    st.rerun()
                    
        with btn_col3:
            st.write("") # placeholder
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col_rhs:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Jeu de test")
        
        if is_test_empty:
            st.error("❌ Le dossier `dataset_annote/test` est vide ou sans labels ! L'évaluation est bloquée.")
            st.info("""
                **Comment résoudre cela ?**
                Allez dans l'onglet **Réentraîner**, puis cliquez sur
                **Préparer / actualiser le split train-val-test**.
                L'app copiera automatiquement une partie de `Data7.off` dans `dataset_annote/test`
                et gardera ces images hors entraînement.
            """)
        else:
            st.markdown(f"📁 **Chemin :** `{os.path.join(dataset_dir, 'test')}`")
            st.markdown(f"🖼️ **Images Test :** <span class='metric-value'>{test_img_count}</span>", unsafe_allow_html=True)
            st.markdown(f"🏷️ **Défauts annotés :** <span class='metric-value'>{test_boxes_count}</span>", unsafe_allow_html=True)
            st.markdown(f"⚪ **Images saines (sans défaut) :** `{test_bg_count}`")
            
            # Show test split distribution
            st.markdown("---")
            st.write("📊 Répartition par classe dans le jeu de Test :")
            available_ids = sorted(test_class_counts.keys()) if test_class_counts else st.session_state.active_model_classes.get("ordered_ids", [])
            chart_test_data = {get_active_class_display(cid): test_class_counts.get(cid, 0) for cid in available_ids}
            st.bar_chart(chart_test_data, horizontal=True, height=130)
            
            # Class threshold warnings
            low_class_warnings = []
            for cid in available_ids:
                cnt = test_class_counts.get(cid, 0)
                if cnt < 5:
                    low_class_warnings.append(f"• **{get_active_class_display(cid)}** ({cnt} ex.)")
            
            if low_class_warnings:
                st.warning("⚠️ **Exemples de test critiques (< 5) :**\n" + "\n".join(low_class_warnings) + "\n\n*Les conclusions sur ces classes peuvent ne pas être fiables à 100%.*")
                
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ COMPILATION D'HISTORIQUE ET COMPARAISON ------------------
    st.markdown("---")
    st.markdown("<h3 style='color:#38BDF8;'>📜 Historique des évaluations</h3>", unsafe_allow_html=True)
    
    history_records = []
    if os.path.exists(history_file_path):
        try:
            with open(history_file_path, "r", encoding="utf-8") as hf:
                history_records = json.load(hf).get("history", [])
        except Exception:
            pass
            
    if not history_records:
        st.info("Aucun modèle n'a encore été évalué. Cliquez sur le bouton 'Évaluer le modèle actuel' ci-dessus pour générer les premières métriques de test.")
    else:
        # History table format
        hist_table_data = []
        for idx, rec in enumerate(history_records):
            m = rec["metrics"]
            meta = rec["metadata"]
            hist_table_data.append({
                "Index": idx,
                "ID": rec["id"],
                "Modèle": rec["model_name"],
                "Date": rec["evaluation_date"],
                "Images Test": rec["test_images_count"],
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f}",
                "F1-score": f"{m['f1']:.3f}",
                "mAP50": f"{m['map50']:.3f}",
                "mAP50-95": f"{m['map50_95']:.3f}",
                "Inférence (ms)": f"{m['inference_speed_ms']:.1f} ms"
            })
            
        df_hist = pd.DataFrame(hist_table_data)
        st.dataframe(df_hist.drop(columns=["Index", "ID"]), use_container_width=True, hide_index=True)
        
        # Select two versions for detailed comparison
        st.markdown("---")
        st.markdown("#### ⚖️ Comparateur de versions de modèles")
        comp_col1, comp_col2, comp_col3 = st.columns([2, 2, 1])
        
        with comp_col1:
            opts_model_a = [f"[{i}] {r['model_name']} - {r['evaluation_date']}" for i, r in enumerate(history_records)]
            selected_opt_a = st.selectbox("Modèle A (Référence / Actuel) :", opts_model_a, key="selectbox_comp_a")
            idx_a = int(selected_opt_a.split(']')[0].replace('[', ''))
            run_a = history_records[idx_a]
            
        with comp_col2:
            opts_model_b = [f"[{i}] {r['model_name']} - {r['evaluation_date']}" for i, r in enumerate(history_records)]
            # default to last run if more than 1
            def_idx_b = len(history_records) - 1 if len(history_records) > 1 else 0
            selected_opt_b = st.selectbox("Modèle B (Nouveau / Réentraîné) :", opts_model_b, index=def_idx_b, key="selectbox_comp_b")
            idx_b = int(selected_opt_b.split(']')[0].replace('[', ''))
            run_b = history_records[idx_b]
            
        with comp_col3:
            st.write("")
            st.write("")
            trigger_comp = st.button("⚖️ Comparer les deux", type="primary", use_container_width=True)
            
        if trigger_comp or st.session_state.get("is_comparing_active", False):
            st.session_state.is_comparing_active = True
            
            # Grab selected runs
            run_a = history_records[idx_a]
            run_b = history_records[idx_b]
            
            metrics_a = run_a["metrics"]
            metrics_b = run_b["metrics"]
            
            # --- A. TABLEAU COMPARATIF GLOBAL ---
            st.markdown("<h3 style='color:#38BDF8;'>📈 Comparaison des Métriques Globales</h3>", unsafe_allow_html=True)
            
            # Deduced TP, FP, FN calculations
            def deduce_counts(p, r, gt):
                if gt <= 0:
                    return 0, 0, 0
                tp = r * gt
                fn = gt - tp
                fp = (tp / p) - tp if p > 0 else 0
                return int(round(tp)), int(round(fp)), int(round(fn))
                
            gt_boxes_a = float(test_boxes_count)
            tp_a, fp_a, fn_a = deduce_counts(metrics_a["precision"], metrics_a["recall"], gt_boxes_a)
            tp_b, fp_b, fn_b = deduce_counts(metrics_b["precision"], metrics_b["recall"], gt_boxes_a)
            
            comparison_rows = [
                ("Precision", metrics_a["precision"], metrics_b["precision"], True),
                ("Recall", metrics_a["recall"], metrics_b["recall"], True),
                ("F1-score", metrics_a["f1"], metrics_b["f1"], True),
                ("mAP50", metrics_a["map50"], metrics_b["map50"], True),
                ("mAP50-95", metrics_a["map50_95"], metrics_b["map50_95"], True),
                ("Faux positifs", fp_a, fp_b, False), # lower is better
                ("Faux négatifs", fn_a, fn_b, False), # lower is better
                ("Inférence par image", metrics_a["inference_speed_ms"], metrics_b["inference_speed_ms"], False) # lower is better
            ]
            
            comparison_table = []
            for title, val_a, val_b, higher_is_better in comparison_rows:
                diff_abs = val_b - val_a
                diff_pct = (diff_abs / val_a * 100) if val_a > 0 else 0.0
                
                # Check status
                if abs(diff_abs) < 0.005:
                    status_text = "⚪ Stable (Évolution négligeable)"
                    status_color = "#94A3B8"
                elif (diff_abs > 0 and higher_is_better) or (diff_abs < 0 and not higher_is_better):
                    status_text = "🟢 Amélioration"
                    status_color = "#10B981"
                else:
                    status_text = "🔴 Dégradation"
                    status_color = "#EF4444"
                    
                comparison_table.append({
                    "Indicateur": title,
                    "Modèle A (Référence)": f"{val_a:.3f}" if isinstance(val_a, float) else str(val_a),
                    "Modèle B (Nouveau)": f"{val_b:.3f}" if isinstance(val_b, float) else str(val_b),
                    "Évolution Absolue": f"{diff_abs:+.3f}" if isinstance(diff_abs, float) else f"{diff_abs:+}",
                    "Évolution (%)": f"{diff_pct:+.1f}%" if val_a > 0 else "N/A",
                    "Statut": status_text,
                    "color": status_color
                })
                
            # Render styled comparison table
            html_rows = ""
            for row in comparison_table:
                html_rows += f"""
                <tr style='border-bottom: 1px solid #1E293D;'>
                    <td style='padding:12px; font-weight:700;'>{row["Indicateur"]}</td>
                    <td style='padding:12px;'>{row["Modèle A (Référence)"]}</td>
                    <td style='padding:12px; font-weight:700;'>{row["Modèle B (Nouveau)"]}</td>
                    <td style='padding:12px; color:{row["color"]}; font-weight:700;'>{row["Évolution Absolue"]}</td>
                    <td style='padding:12px; color:{row["color"]}; font-weight:700;'>{row["Évolution (%)"]}</td>
                    <td style='padding:12px; color:{row["color"]}; font-weight:800;'>{row["Statut"]}</td>
                </tr>
                """
                
            st.markdown(f"""
                <table style='width:100%; border-collapse: collapse; background-color: #131926; border-radius: 8px; overflow: hidden;'>
                    <thead>
                        <tr style='background-color: #1E293B; color: #FFF; text-align: left;'>
                            <th style='padding:12px;'>Indicateur</th>
                            <th style='padding:12px;'>Modèle A ({run_a["model_name"]})</th>
                            <th style='padding:12px;'>Modèle B ({run_b["model_name"]})</th>
                            <th style='padding:12px;'>Évolution</th>
                            <th style='padding:12px;'>Évolution (%)</th>
                            <th style='padding:12px;'>Statut</th>
                        </tr>
                    </thead>
                    <tbody>
                        {html_rows}
                    </tbody>
                </table>
            """, unsafe_allow_html=True)
            
            # --- B. TABLEAU PAR CLASSE ---
            st.markdown("<h3 style='color:#38BDF8; margin-top:25px;'>📋 Performances détaillées par Classe</h3>", unsafe_allow_html=True)
            
            class_comparison_table = []
            ordered_ids_eval = st.session_state.active_model_classes.get("ordered_ids", [])
            if not ordered_ids_eval:
                ordered_ids_eval = list(range(len(class_names)))

            def fetch_per_class(metrics_dict, cid):
                per_cls = metrics_dict.get("per_class", {})
                key_candidates = [
                    str(cid),
                    get_active_class_display(cid),
                    get_active_class_name(cid),
                    class_names[cid] if cid < len(class_names) else None
                ]
                for candidate in key_candidates:
                    if candidate and candidate in per_cls:
                        return per_cls[candidate]
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "map50": 0.0, "map50_95": 0.0}

            for cid in ordered_ids_eval:
                display_name = get_active_class_display(cid)
                cls_a = fetch_per_class(metrics_a, cid)
                cls_b = fetch_per_class(metrics_b, cid)

                annot_count = test_class_counts.get(cid, 0)
                diff_map50 = cls_b.get("map50", 0.0) - cls_a.get("map50", 0.0)

                if annot_count < 5:
                    conclusion = "⚠️ Test insuffisant"
                    c_color = "#F59E0B"
                elif abs(diff_map50) < 0.02:
                    conclusion = "⚪ Stable"
                    c_color = "#94A3B8"
                elif diff_map50 > 0.02:
                    conclusion = "🟢 En nette amélioration"
                    c_color = "#10B981"
                else:
                    conclusion = "🔴 En baisse"
                    c_color = "#EF4444"

                class_comparison_table.append({
                    "Classe": f"ID {cid} — {display_name}",
                    "Test Annotations": annot_count,
                    "P. A": f"{cls_a.get('precision', 0.0):.2f}",
                    "P. B": f"{cls_b.get('precision', 0.0):.2f}",
                    "R. A": f"{cls_a.get('recall', 0.0):.2f}",
                    "R. B": f"{cls_b.get('recall', 0.0):.2f}",
                    "mAP50. A": f"{cls_a.get('map50', 0.0):.2f}",
                    "mAP50. B": f"{cls_b.get('map50', 0.0):.2f}",
                    "mAP50-95. A": f"{cls_a.get('map50_95', 0.0):.2f}",
                    "mAP50-95. B": f"{cls_b.get('map50_95', 0.0):.2f}",
                    "Conclusion": conclusion,
                    "color": c_color
                })
                
            html_class_rows = ""
            for row in class_comparison_table:
                html_class_rows += f"""
                <tr style='border-bottom: 1px solid #1E293D;'>
                    <td style='padding:10px; font-weight:700;'>{row["Classe"]}</td>
                    <td style='padding:10px; text-align:center;'>{row["Test Annotations"]}</td>
                    <td style='padding:10px;'>{row["P. A"]} ➜ <strong>{row["P. B"]}</strong></td>
                    <td style='padding:10px;'>{row["R. A"]} ➜ <strong>{row["R. B"]}</strong></td>
                    <td style='padding:10px;'>{row["mAP50. A"]} ➜ <strong>{row["mAP50. B"]}</strong></td>
                    <td style='padding:10px;'>{row["mAP50-95. A"]} ➜ <strong>{row["mAP50-95. B"]}</strong></td>
                    <td style='padding:10px; color:{row["color"]}; font-weight:800;'>{row["Conclusion"]}</td>
                </tr>
                """
                
            st.markdown(f"""
                <table style='width:100%; border-collapse: collapse; background-color: #131926; border-radius: 8px; overflow: hidden;'>
                    <thead>
                        <tr style='background-color: #1E293B; color: #FFF; text-align: left;'>
                            <th style='padding:10px;'>Type de défaut (Classe)</th>
                            <th style='padding:10px; text-align:center;'>Annotations Test</th>
                            <th style='padding:10px;'>Precision (A ➜ B)</th>
                            <th style='padding:10px;'>Recall (A ➜ B)</th>
                            <th style='padding:10px;'>mAP50 (A ➜ B)</th>
                            <th style='padding:10px;'>mAP50-95 (A ➜ B)</th>
                            <th style='padding:10px;'>Suivi / Conclusion</th>
                        </tr>
                    </thead>
                    <tbody>
                        {html_class_rows}
                    </tbody>
                </table>
            """, unsafe_allow_html=True)
            
            # --- C. AFFICHAGE DES IMAGES DE DIAGNOSTICS (Matrices de confusion / Courbes) ---
            st.markdown("<h3 style='color:#38BDF8; margin-top:25px;'>🖼️ Courbes et Matrices de Confusion de l'évaluation</h3>", unsafe_allow_html=True)
            
            col_plot_a, col_plot_b = st.columns(2)
            with col_plot_a:
                st.markdown(f"**Modèle A ({run_a['model_name']}) :**")
                path_a_dir = os.path.join(eval_runs_base, run_a["id"])
                
                # Render Confusion matrix if it exists
                conf_a_path = os.path.join(path_a_dir, "confusion_matrix.png")
                if os.path.exists(conf_a_path):
                    st.image(conf_a_path, caption="Confusion Matrix - Modèle A", use_container_width=True)
                else:
                    st.info("Matrice de confusion indisponible.")
                    
                pr_a_path = os.path.join(path_a_dir, "PR_curve.png")
                if os.path.exists(pr_a_path):
                    st.image(pr_a_path, caption="Precision-Recall Curve - Modèle A", use_container_width=True)
                else:
                    pr_a_path_2 = os.path.join(path_a_dir, "P_curve.png")
                    if os.path.exists(pr_a_path_2):
                        st.image(pr_a_path_2, caption="Precision Curve - Modèle A", use_container_width=True)
            
            with col_plot_b:
                st.markdown(f"**Modèle B ({run_b['model_name']}) :**")
                path_b_dir = os.path.join(eval_runs_base, run_b["id"])
                
                conf_b_path = os.path.join(path_b_dir, "confusion_matrix.png")
                if os.path.exists(conf_b_path):
                    st.image(conf_b_path, caption="Confusion Matrix - Modèle B", use_container_width=True)
                else:
                    st.info("Matrice de confusion indisponible.")
                    
                pr_b_path = os.path.join(path_b_dir, "PR_curve.png")
                if os.path.exists(pr_b_path):
                    st.image(pr_b_path, caption="Precision-Recall Curve - Modèle B", use_container_width=True)
                else:
                    pr_b_path_2 = os.path.join(path_b_dir, "P_curve.png")
                    if os.path.exists(pr_b_path_2):
                        st.image(pr_b_path_2, caption="Precision Curve - Modèle B", use_container_width=True)

            # --- D. ANALYSIS OF DATASET SPLITS (Vérifications globales et cross-splits) ---
            st.markdown("<h3 style='color:#38BDF8; margin-top:25px;'>📁 Tableau d'Analyse du Dataset global</h3>", unsafe_allow_html=True)
            
            # Prepare Split verification rows
            split_rows_html = ""
            for split in ["train", "val", "test"]:
                sp_st = dataset_stats.get(split, {})
                sp_img_cnt = sp_st.get("images_count", 0)
                sp_lbl_cnt = sp_st.get("labels_count", 0)
                sp_box_cnt = sp_st.get("boxes_count", 0)
                sp_missing_lbl = sp_st.get("missing_labels", 0)
                sp_orphaned_lbl = sp_st.get("missing_images", 0)
                
                # Check cross split duplicates for this split
                dups = len(cross_split_duplicates.get(split, []))
                dup_text = f"🟢 Aucun" if dups == 0 else f"🔴 {dups} images dupliquées"
                
                split_rows_html += f"""
                <tr style='border-bottom: 1px solid #1E293D;'>
                    <td style='padding:10px; font-weight:700; text-transform: capitalize;'>{split}</td>
                    <td style='padding:10px;'>{sp_img_cnt}</td>
                    <td style='padding:10px;'>{sp_lbl_cnt}</td>
                    <td style='padding:10px;'>{sp_box_cnt}</td>
                    <td style='padding:10px; color:{"#EF4444" if sp_missing_lbl > 0 else "#10B981"}; font-weight:700;'>{sp_missing_lbl}</td>
                    <td style='padding:10px; color:{"#EF4444" if sp_orphaned_lbl > 0 else "#10B981"}; font-weight:700;'>{sp_orphaned_lbl}</td>
                    <td style='padding:10px;'>{dup_text}</td>
                </tr>
                """
                
            st.markdown(f"""
                <table style='width:100%; border-collapse: collapse; background-color: #131926; border-radius: 8px; overflow: hidden;'>
                    <thead>
                        <tr style='background-color: #1E293B; color: #FFF; text-align: left;'>
                            <th style='padding:10px;'>Split Dataset</th>
                            <th style='padding:10px;'>Images Totales</th>
                            <th style='padding:10px;'>Labels Présents</th>
                            <th style='padding:10px;'>Annotations BBoxes</th>
                            <th style='padding:10px;'>Images sans Label (Orphelines)</th>
                            <th style='padding:10px;'>Labels sans Image (Orphelins)</th>
                            <th style='padding:10px;'>Doublons croisés (entre splits)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {split_rows_html}
                    </tbody>
                </table>
            """, unsafe_allow_html=True)
            
            # --- E. TRAINING CURVES (results.csv) ---
            csv_best_name = f"results_{run_b['model_name'].replace('.pt', '')}.csv"
            csv_best_path = os.path.join(base_dir, csv_best_name)
            
            if os.path.exists(csv_best_path):
                st.markdown("<h3 style='color:#38BDF8; margin-top:25px;'>📈 Courbes d'Entraînement du Modèle Réentraîné</h3>", unsafe_allow_html=True)
                try:
                    df_curves = pd.read_csv(csv_best_path)
                    # Strip spaces in column headers
                    df_curves.columns = [c.strip() for c in df_curves.columns]
                    
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        # Box Loss
                        loss_cols = [c for c in ["train/box_loss", "val/box_loss"] if c in df_curves.columns]
                        if loss_cols:
                            st.line_chart(df_curves[loss_cols], use_container_width=True)
                            st.caption("Évolution de la Box Loss (Train vs Val)")
                            
                        # Cls Loss
                        cls_loss_cols = [c for c in ["train/cls_loss", "val/cls_loss"] if c in df_curves.columns]
                        if cls_loss_cols:
                            st.line_chart(df_curves[cls_loss_cols], use_container_width=True)
                            st.caption("Évolution de la Classification Loss (Train vs Val)")
                            
                    with tc2:
                        # mAPs
                        map_cols = [c for c in ["metrics/mAP50(B)", "metrics/mAP50-95(B)"] if c in df_curves.columns]
                        if map_cols:
                            st.line_chart(df_curves[map_cols], use_container_width=True)
                            st.caption("Métrique mAP50 et mAP50-95 par Époque")
                            
                        # Precision / Recall
                        pr_cols = [c for c in ["metrics/precision(B)", "metrics/recall(B)"] if c in df_curves.columns]
                        if pr_cols:
                            st.line_chart(df_curves[pr_cols], use_container_width=True)
                            st.caption("Précision et Recall globaux par Époque")
                            
                except Exception as ex_c:
                    st.warning(f"Impossible de tracer les courbes d'entraînement : {ex_c}")

            # --- F. ANALYSE EXPERTE ET RECOMMANDATIONS DETERMINISTES ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='background-color:#0F172A; border: 2px solid #38BDF8; padding: 25px; border-radius: 12px;'>", unsafe_allow_html=True)
            st.markdown("<h2 style='color:#38BDF8; margin-top:0px; display:flex; align-items:center;'>🧠 Analyse experte et Recommandations</h2>", unsafe_allow_html=True)
            
            # Setup deterministic analysis
            positives = []
            problems = []
            recommendations = []
            
            # Grab dataset properties
            train_stats = dataset_stats.get("train", {})
            train_box_counts = train_stats.get("class_counts", {i: 0 for i in range(len(class_names))})
            train_boxes_sum = train_stats.get("boxes_count", 0)
            
            # Evaluate model progress positives
            map_diff = metrics_b["map50_95"] - metrics_a["map50_95"]
            rec_diff = metrics_b["recall"] - metrics_a["recall"]
            
            if map_diff > 0.02:
                positives.append(f"Le nouveau modèle améliore substantiellement le mAP50-95 global, passant de `{metrics_a['map50_95']:.2f}` à `{metrics_b['map50_95']:.2f}` (+{map_diff:.1%}).")
            if rec_diff > 0.02:
                positives.append(f"Le Recall progresse fortement de `{metrics_a['recall']:.2f}` à `{metrics_b['recall']:.2f}` : l'IA oublie moins de dommages réels.")
            if len(positives) == 0:
                positives.append("Les performances générales du modèle restent globalement équivalentes sur ce jeu de test.")
                
            # Rule A: Probable Lack of Data
            rare_train_classes = []
            for i, name in enumerate(class_names):
                if train_box_counts.get(i, 0) < 15:
                    rare_train_classes.append(class_translations[name])
                    
            if metrics_b["map50_95"] < 0.45 and metrics_b["recall"] < 0.50 and len(rare_train_classes) > 0:
                problems.append("Le modèle semble manquer d'exemples d'entraînement pour capturer correctement toutes les classes.")
                recommendations.append({
                    "priority": "Priorité haute 🔴",
                    "text": f"Collecter et annoter au moins 80 nouveaux exemples pour les classes rares d'entraînement : {', '.join(rare_train_classes)}."
                })
                
            # Rule B: Class Imbalance
            avg_train_annotations = train_boxes_sum / len(class_names) if len(class_names) > 0 else 0
            imbalanced_classes = []
            if avg_train_annotations > 0:
                for i, name in enumerate(class_names):
                    c_cnt = train_box_counts.get(i, 0)
                    if c_cnt < (0.15 * avg_train_annotations):
                        imbalanced_classes.append(class_translations[name])
                        
            if imbalanced_classes:
                problems.append(f"Déséquilibre sévère de classe détecté : les classes `{', '.join(imbalanced_classes)}` sont sous-représentées dans le jeu d'entraînement.")
                recommendations.append({
                    "priority": "Priorité haute 🔴",
                    "text": "Ajouter des exemples ciblés des classes sous-représentées pour rééquilibrer le dataset avant le prochain cycle d'entraînement."
                })
                
            # Rule C & D: Overfitting / Lack of Epochs (Read results.csv if exists)
            has_overfit_signal = False
            has_more_epochs_signal = False
            if os.path.exists(csv_best_path):
                try:
                    df_curves = pd.read_csv(csv_best_path)
                    df_curves.columns = [c.strip() for c in df_curves.columns]
                    if "val/box_loss" in df_curves.columns and len(df_curves) > 5:
                        val_losses = df_curves["val/box_loss"].tolist()
                        train_losses = df_curves["train/box_loss"].tolist()
                        
                        # Overfitting check: train loss decreases but val loss increases at the end
                        recent_val_avg = sum(val_losses[-3:]) / 3
                        prev_val_avg = sum(val_losses[-8:-3]) / 5 if len(val_losses) >= 8 else val_losses[0]
                        
                        if recent_val_avg > prev_val_avg * 1.05:
                            has_overfit_signal = True
                        else:
                            # Not overfitting, check if still learning
                            map_vals = df_curves["metrics/mAP50-95(B)"].tolist() if "metrics/mAP50-95(B)" in df_curves.columns else []
                            if map_vals and map_vals[-1] > map_vals[-3] + 0.005:
                                has_more_epochs_signal = True
                except Exception:
                    pass
                    
            if has_overfit_signal:
                problems.append("Risque de surapprentissage (overfitting) détecté : la perte de validation remonte en fin d'entraînement.")
                recommendations.append({
                    "priority": "Priorité moyenne 🟡",
                    "text": "Intégrer du Early Stopping à l'entraînement, réduire le nombre d'époques effectives, et ajouter des augmentations d'images pour régulariser le modèle."
                })
            elif has_more_epochs_signal:
                recommendations.append({
                    "priority": "Priorité faible 🟢",
                    "text": "L'entraînement progressait encore à la fin. Tester un réentraînement avec 15 à 30 époques de plus tout en surveillant la validation."
                })
                
            # Rule E: Too many False Positives
            if metrics_b["precision"] < 0.55 and metrics_b["recall"] > 0.70:
                problems.append("Le modèle génère trop de fausses alertes (Precision faible, Recall élevé).")
                recommendations.append({
                    "priority": "Priorité haute 🔴",
                    "text": "Ajouter au moins 50 images négatives saines (sans défaut), comprenant des carrosseries propres, des reflets et des poussières pour apprendre à l'IA à ignorer ces éléments."
                })
                
            # Rule F: Too many False Negatives (Missed detections)
            if metrics_b["recall"] < 0.50:
                problems.append("Le modèle oublie trop de dommages réels présents (Recall faible).")
                recommendations.append({
                    "priority": "Priorité haute 🔴",
                    "text": "Collecter des photos de défauts complexes sous différents angles et éclairages (par ex: micro-rayures, légères bosses sur véhicules sombres)."
                })
                
            # Rule G: Imprecise bounding boxes
            if (metrics_b["map50"] - metrics_b["map50_95"]) > 0.25:
                problems.append("Les défauts sont localisés de manière approximative (Le mAP50 est élevé mais le mAP50-95 s'effondre).")
                recommendations.append({
                    "priority": "Priorité moyenne 🟡",
                    "text": "Améliorer la rigueur des annotations manuelles. Les rectangles doivent épouser parfaitement les contours physiques des rayures ou bosses sans déborder."
                })
                
            # Rule H: Unreliable test set
            classes_under_test_threshold = []
            for i, name in enumerate(class_names):
                if test_class_counts.get(i, 0) < 5:
                    classes_under_test_threshold.append(class_translations[name])
                    
            if test_img_count < 20 or len(classes_under_test_threshold) > 0:
                problems.append(f"Le jeu de test actuel est insuffisant ou déséquilibré pour évaluer de manière fiable les classes : {', '.join(classes_under_test_threshold)}.")
                recommendations.append({
                    "priority": "Priorité haute 🔴",
                    "text": "Enrichir le jeu de test fixe d'au moins 30 nouvelles photos annotées comprenant toutes les catégories de dommages."
                })
                
            # Cross-split duplicate warnings
            cross_dups_sum = sum(len(cross_split_duplicates.get(split, [])) for split in ["train", "val", "test"])
            if cross_dups_sum > 0:
                problems.append(f"Alerte de fuite de données : {cross_dups_sum} images dupliquées ont été identifiées entre les splits (ex: test et train).")
                recommendations.append({
                    "priority": "Priorité haute 🔴",
                    "text": "Nettoyer le dataset pour s'assurer qu'aucune image présente dans 'test' n'apparaisse dans 'train' ou 'val'."
                })

            # Display sections
            st.markdown("### 👍 Points forts du nouveau modèle")
            for pos in positives:
                st.write(f"✓ {pos}")
                
            if problems:
                st.markdown("### ⚠️ Problèmes et limites identifiés")
                for pr in problems:
                    st.write(f"⚠ {pr}")
                    
            st.markdown("### 📋 Plan de recommandations prioritaires")
            # Sort recommendations by high priority first
            sorted_recs = sorted(recommendations, key=lambda x: "🔴" not in x["priority"])
            for rec in sorted_recs:
                st.markdown(f"- **[{rec['priority']}]** {rec['text']}")
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            # --- G. ACTIONS DIRECTES DE PRODUCTION (Bouton Définir comme Actif) ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🚀 Déploiement en Production")
            
            is_b_active = os.path.basename(active_model_path) == run_b["model_name"]
            
            if is_b_active:
                st.info(f"🏆 Le modèle **{run_b['model_name']}** est déjà sélectionné dans l'application.")
            else:
                st.warning(f"Le modèle de référence actif est actuellement **{os.path.basename(active_model_path)}**.")
                deploy_col1, deploy_col2 = st.columns([2, 3])
                with deploy_col1:
                    deploy_btn = st.button(f"🚀 Remplacer best_2.pt par {run_b['model_name']}", type="primary", use_container_width=True)
                    if deploy_btn:
                        # Copy weight file to best_2.pt, the single reference model.
                        src_model_pt = run_b["model_path"]
                        dst_model_pt = os.path.join(base_dir, "best_2.pt")
                        
                        if os.path.exists(src_model_pt):
                            # Backup
                            if os.path.exists(dst_model_pt):
                                shutil.copy(dst_model_pt, os.path.join(base_dir, f"best_2_backup_before_deploy_{int(time.time())}.pt"))
                            shutil.copy(src_model_pt, dst_model_pt)
                            
                            st.cache_resource.clear()
                            st.balloons()
                            st.success(f"🏆 Modèle {run_b['model_name']} déployé avec succès comme nouvelle référence `best_2.pt` !")
                            st.rerun()
                        else:
                            st.error(f"Fichier source introuvable pour copie : {src_model_pt}")
                with deploy_col2:
                    st.write("Ce bouton copie le modèle choisi pour remplacer `best_2.pt`. Une sauvegarde de l'ancien `best_2.pt` est créée automatiquement.")

# --- TAB 5: CLASS HARMONIZATION STUDIO ---
if selected_section == NAV_SECTIONS[4]:
    st.markdown("<h2 style='color:#38BDF8;'>🔍 Harmoniser les classes</h2>", unsafe_allow_html=True)
    st.markdown("""
        Alignez les identifiants de classes entre vos annotations, votre modèle et les datasets externes.
        Le format cible suit `best_2.pt` : `crack`, `dent`, `glass shatter`, `lamp broken`, `scratch`, `tire flat`.
    """)
    
    import yaml
    import json
    import os
    import shutil
    import time
    from datetime import datetime
    import pandas as pd
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    final_yaml_path = os.path.join(base_dir, "final_classes.yaml")
    
    # 1. Load Cible Classes (final_classes.yaml)
    final_classes = dict(FINAL_DATASET_CLASSES)
    if os.path.exists(final_yaml_path):
        try:
            with open(final_yaml_path, "r", encoding="utf-8") as f_y:
                y_data = yaml.safe_load(f_y)
                if y_data and "names" in y_data:
                    final_classes = {int(k): v for k, v in y_data["names"].items()}
        except Exception:
            pass
            
    # Display reference classes
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Classes finales")
    st.write("Ces classes servent de référence pour le dataset final.")
    
    ref_col1, ref_col2, ref_col3 = st.columns(3)
    for cid, cname in final_classes.items():
        with [ref_col1, ref_col2, ref_col3][cid % 3]:
            st.markdown(f"""
                <div style='background-color: #1E293B; border-left: 5px solid {['#5B9CF6', '#F97066', '#63D0A8'][cid % 3]}; padding: 10px 15px; border-radius: 4px; margin-bottom: 8px;'>
                    <strong style='font-size: 1.1rem; color: #FFF;'>ID {cid} : {cname}</strong>
                    <div style='color: #94A3B8; font-size: 0.8rem;'>Label standard final</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Analyze sources
    st.markdown("### 🔬 Sources détectées")
    
    # A. Active model classes. Reuse the already loaded model metadata instead of
    # loading best_2.pt again on every harmonization render.
    model_classes = {
        int(k): str(v)
        for k, v in st.session_state.get("active_model_classes", {}).get("id_to_name", {}).items()
    }
            
    # B. Site classes (the dropdown list defined in app.py)
    site_classes = {i: name for i, name in enumerate(class_names)}
    
    # C. Existing labels occurrences count
    labels_dir = os.path.join(base_dir, "Data7.off", "labels")
    lbl_counts = {}
    if os.path.exists(labels_dir):
        try:
            for f in os.listdir(labels_dir):
                if f.endswith('.txt'):
                    with open(os.path.join(labels_dir, f), "r") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if parts:
                                cid = int(parts[0])
                                lbl_counts[cid] = lbl_counts.get(cid, 0) + 1
        except Exception:
            pass
            
    # D. External dataset path
    default_ext_path = r"C:\Users\p134929\Downloads\job_3629957_annotations_2026_02_16_15_10_12_ultralytics yolo detection 1.0"
    ext_path = st.text_input("📁 Chemin du dataset YOLO externe à analyser :", value=default_ext_path)
    
    ext_classes = {}
    if os.path.exists(ext_path):
        ext_yaml = os.path.join(ext_path, "data.yaml")
        if os.path.exists(ext_yaml):
            try:
                with open(ext_yaml, "r") as ef:
                    ey = yaml.safe_load(ef)
                    if ey and "names" in ey:
                        ext_classes = {int(k): v for k, v in ey["names"].items()}
            except Exception:
                pass
                
    # E. Build comparative dataframe
    st.markdown("#### ⚖️ Correspondances proposées")
    
    comparison_data = []
    
    # Model rows
    for cid, name in model_classes.items():
        # Match with final target
        matched_id = "Incompatible"
        matched_name = "-"
        status = "🔴 Incompatible"
        for fid, fname in final_classes.items():
            if fname.lower() == name.lower():
                matched_id = str(fid)
                matched_name = fname
                status = "🟢 Traduction possible"
                if fid == cid:
                    status = "🟢 Compatible"
                break
                
        comparison_data.append({
            "Source": "Modèle Actuel",
            "ID Source": cid,
            "Nom Source": name,
            "ID Final Proposé": matched_id,
            "Classe Finale": matched_name,
            "Statut": status
        })
        
    # Site rows
    for cid, name in site_classes.items():
        matched_id = "Ignorer / Manuel"
        matched_name = "-"
        status = "🟡 À ignorer"
        occ = lbl_counts.get(cid, 0)
        
        for fid, fname in final_classes.items():
            if fname.lower() == name.lower():
                matched_id = str(fid)
                matched_name = fname
                status = "🔴 À remapper"
                if fid == cid:
                    status = "🟢 Compatible"
                break
                
        if matched_id == "Ignorer / Manuel" and occ > 0:
            status = "⚠️ Ambigu (Annotations présentes !)"
            
        comparison_data.append({
            "Source": f"Site ({occ} annots)",
            "ID Source": cid,
            "Nom Source": name,
            "ID Final Proposé": matched_id,
            "Classe Finale": matched_name,
            "Statut": status
        })
        
    # External dataset rows
    for cid, name in ext_classes.items():
        matched_id = "Ignorer / Manuel"
        matched_name = "-"
        status = "🟡 À ignorer"
        
        for fid, fname in final_classes.items():
            if fname.lower() == name.lower():
                matched_id = str(fid)
                matched_name = fname
                status = "🟢 Traduction possible"
                if fid == cid:
                    status = "🟢 Compatible"
                break
                
        comparison_data.append({
            "Source": "Dataset Externe",
            "ID Source": cid,
            "Nom Source": name,
            "ID Final Proposé": matched_id,
            "Classe Finale": matched_name,
            "Statut": status
        })
        
    if comparison_data:
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune source détectée à analyser.")

    # 3. Validation forms for mapping overrides
    st.markdown("---")
    st.markdown("### 🛠️ Mapping final")
    st.write("Choisissez la destination de chaque classe source. Les classes à vérifier seront isolées dans un dossier séparé.")
    
    col_f1, col_f2 = st.columns(2)
    
    final_options = list(final_classes.values()) + ["Ignorer cette classe", "À vérifier manuellement"]
    
    site_mapping = {}
    ext_mapping = {}
    
    with col_f1:
        st.markdown("**Mapping de l'interface / annotations du Site :**")
        for cid, name in site_classes.items():
            occ = lbl_counts.get(cid, 0)
            if occ == 0 and cid not in final_classes:
                # default ignore for unused non-core classes
                def_idx = final_options.index("Ignorer cette classe")
            elif name in final_classes.values():
                def_idx = final_options.index(name)
            else:
                def_idx = final_options.index("À vérifier manuellement")
                
            sel_opt = st.selectbox(
                f"Classe du Site #{cid} : `{name}` ({occ} annots)",
                final_options,
                index=def_idx,
                key=f"site_map_sel_{cid}"
            )
            site_mapping[cid] = sel_opt
            
    with col_f2:
        st.markdown("**Mapping du Dataset Externe (YOLO à importer) :**")
        if ext_classes:
            for cid, name in ext_classes.items():
                if name in final_classes.values():
                    def_idx = final_options.index(name)
                else:
                    def_idx = final_options.index("Ignorer cette classe")
                    
                sel_opt = st.selectbox(
                    f"Dataset Externe #{cid} : `{name}`",
                    final_options,
                    index=def_idx,
                    key=f"ext_map_sel_{cid}"
                )
                ext_mapping[cid] = sel_opt
        else:
            st.warning("Aucun dataset externe valide n'est actuellement détecté. Renseignez un chemin contenant un `data.yaml` ci-dessus.")

    # 4. Conversion execution panel
    st.markdown("---")
    st.markdown("### 📦 Générer le dataset harmonisé")
    st.write("Crée une sauvegarde, convertit les labels, puis écrit le résultat dans `dataset_final_harmonise`.")
    
    # Warnings checking
    has_ambiguous_site_classes = any([v == "À vérifier manuellement" for k, v in site_mapping.items() if lbl_counts.get(k, 0) > 0])
    
    if has_ambiguous_site_classes:
        st.info("💡 **Remarque :** Certaines annotations présentes dans vos fichiers seront routées vers le sous-dossier `a_verifier_manuellement/` car elles possèdent des classes ambiguës.")
        
    trigger_harmonization = st.button("⚡ Exécuter l'harmonisation du Dataset", type="primary", use_container_width=True)
    
    if trigger_harmonization:
        try:
            with st.status("🛠️ Lancement du pipeline d'harmonisation des données...", expanded=True) as status:
                # Create Backup of Data7.off labels
                status.update(label="📁 Création d'une sauvegarde de sécurité...", state="running")
                backup_base = os.path.join(base_dir, "backups_original_annotations")
                os.makedirs(backup_base, exist_ok=True)
                
                timestamp = int(time.time())
                backup_dir = os.path.join(backup_base, f"backup_annotations_{timestamp}")
                
                if os.path.exists(labels_dir) and len(os.listdir(labels_dir)) > 0:
                    shutil.copytree(labels_dir, backup_dir)
                    status.write(f"💾 Sauvegarde créée avec succès sous `{os.path.basename(backup_dir)}` !")
                else:
                    status.write("⏳ Aucune annotation existante dans Data7.off/labels à sauvegarder.")
                    
                # Create Output Harmonized Directories
                status.update(label="📦 Initialisation des dossiers harmonisés...", state="running")
                out_dir = os.path.join(base_dir, "dataset_final_harmonise")
                out_images = os.path.join(out_dir, "images")
                out_labels = os.path.join(out_dir, "labels")
                
                manual_images = os.path.join(out_dir, "a_verifier_manuellement", "images")
                manual_labels = os.path.join(out_dir, "a_verifier_manuellement", "labels")
                
                # Delete any old version to prevent overlap
                if os.path.exists(out_dir):
                    try:
                        shutil.rmtree(out_dir)
                    except Exception:
                        pass
                        
                os.makedirs(out_images, exist_ok=True)
                os.makedirs(out_labels, exist_ok=True)
                os.makedirs(manual_images, exist_ok=True)
                os.makedirs(manual_labels, exist_ok=True)
                
                # Process local site annotations
                status.update(label="🔄 Analyse et remapping des fichiers d'annotations locaux...", state="running")
                
                img_src_dir = os.path.join(base_dir, "Data7.off", "images")
                
                total_local_files = 0
                converted_annots = 0
                unchanged_annots = 0
                manual_files_count = 0
                ignored_annots = 0
                
                # Build dict option map
                # key is class ID, value is final ID (int) or None or "MANUAL"
                map_lookup = {}
                for k, v in site_mapping.items():
                    if v in final_classes.values():
                        # Find the index
                        for f_id, f_name in final_classes.items():
                            if f_name == v:
                                map_lookup[k] = f_id
                                break
                    elif v == "À vérifier manuellement":
                        map_lookup[k] = "MANUAL"
                    else:
                        map_lookup[k] = "IGNORE"
                        
                if os.path.exists(labels_dir):
                    lbl_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
                    total_local_files = len(lbl_files)
                    
                    for lf in lbl_files:
                        lbl_path_in = os.path.join(labels_dir, lf)
                        
                        # Check corresponding image
                        img_name_no_ext, _ = os.path.splitext(lf)
                        img_found = None
                        if os.path.exists(img_src_dir):
                            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                                if os.path.exists(os.path.join(img_src_dir, f"{img_name_no_ext}{ext}")):
                                    img_found = f"{img_name_no_ext}{ext}"
                                    break
                                    
                        # Read and parse labels
                        lines_to_keep_harmonized = []
                        lines_to_keep_manual = []
                        is_manual_required = False
                        
                        with open(lbl_path_in, "r") as l_in:
                            for line in l_in:
                                parts = line.strip().split()
                                if parts:
                                    cid_in = int(parts[0])
                                    coords = parts[1:]
                                    
                                    action = map_lookup.get(cid_in, "IGNORE")
                                    
                                    if action == "MANUAL":
                                        is_manual_required = True
                                        lines_to_keep_manual.append(f"{cid_in} " + " ".join(coords))
                                    elif isinstance(action, int):
                                        lines_to_keep_harmonized.append(f"{action} " + " ".join(coords))
                                        if action == cid_in:
                                            unchanged_annots += 1
                                        else:
                                            converted_annots += 1
                                    else:
                                        ignored_annots += 1
                                        
                        # Write the corresponding labels and copy images
                        if is_manual_required:
                            manual_files_count += 1
                            # All boxes go to manual check to avoid partial split
                            shutil.copy(lbl_path_in, os.path.join(manual_labels, lf))
                            if img_found:
                                shutil.copy(os.path.join(img_src_dir, img_found), os.path.join(manual_images, img_found))
                        else:
                            # Save to harmonized
                            with open(os.path.join(out_labels, lf), "w") as l_out:
                                l_out.write("\n".join(lines_to_keep_harmonized) + "\n")
                            if img_found:
                                shutil.copy(os.path.join(img_src_dir, img_found), os.path.join(out_images, img_found))
                                
                # Parse external dataset if path provided and valid
                total_ext_files_copied = 0
                if ext_classes and os.path.exists(ext_path):
                    status.update(label="📂 Import et harmonisation du dataset externe...", state="running")
                    
                    ext_labels_dir = os.path.join(ext_path, "labels")
                    ext_images_dir = os.path.join(ext_path, "images")
                    
                    # check fallback or direct txt list
                    if not os.path.exists(ext_labels_dir):
                        # try root path
                        ext_labels_dir = ext_path
                        ext_images_dir = ext_path
                        
                    ext_map_lookup = {}
                    for k, v in ext_mapping.items():
                        if v in final_classes.values():
                            for f_id, f_name in final_classes.items():
                                if f_name == v:
                                    ext_map_lookup[k] = f_id
                                    break
                        elif v == "À vérifier manuellement":
                            ext_map_lookup[k] = "MANUAL"
                        else:
                            ext_map_lookup[k] = "IGNORE"
                            
                    if os.path.exists(ext_labels_dir):
                        ext_lbl_files = [f for f in os.listdir(ext_labels_dir) if f.endswith('.txt')]
                        
                        for lf in ext_lbl_files:
                            lbl_path_in = os.path.join(ext_labels_dir, lf)
                            img_name_no_ext, _ = os.path.splitext(lf)
                            
                            img_found = None
                            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                                if os.path.exists(os.path.join(ext_images_dir, f"{img_name_no_ext}{ext}")):
                                    img_found = f"{img_name_no_ext}{ext}"
                                    break
                                    
                            lines_to_keep_harmonized = []
                            lines_to_keep_manual = []
                            is_manual_required = False
                            
                            with open(lbl_path_in, "r") as l_in:
                                for line in l_in:
                                    parts = line.strip().split()
                                    if parts:
                                        cid_in = int(parts[0])
                                        coords = parts[1:]
                                        
                                        action = ext_map_lookup.get(cid_in, "IGNORE")
                                        
                                        if action == "MANUAL":
                                            is_manual_required = True
                                            lines_to_keep_manual.append(f"{cid_in} " + " ".join(coords))
                                        elif isinstance(action, int):
                                            lines_to_keep_harmonized.append(f"{action} " + " ".join(coords))
                                            converted_annots += 1
                                        else:
                                            ignored_annots += 1
                                            
                            if is_manual_required:
                                manual_files_count += 1
                                shutil.copy(lbl_path_in, os.path.join(manual_labels, lf))
                                if img_found:
                                    shutil.copy(os.path.join(ext_images_dir, img_found), os.path.join(manual_images, img_found))
                            else:
                                with open(os.path.join(out_labels, lf), "w") as l_out:
                                    l_out.write("\n".join(lines_to_keep_harmonized) + "\n")
                                if img_found:
                                    shutil.copy(os.path.join(ext_images_dir, img_found), os.path.join(out_images, img_found))
                                    total_ext_files_copied += 1
                                    
                # Create YAML files in target folder
                names_yaml_content = "\n".join([f"  {i}: \"{name}\"" for i, name in final_classes.items()])
                yaml_data_content = f"""
path: "{out_dir.replace('\\', '/')}"
train: "images"
val: "images"

names:
{names_yaml_content}
"""
                with open(os.path.join(out_dir, "final_classes.yaml"), "w", encoding="utf-8") as fy_out:
                    fy_out.write(yaml_data_content.strip())
                    
                # Save mapping applied json
                mapping_applied = {
                    "site_mapping": {str(k): v for k, v in site_mapping.items()},
                    "ext_mapping": {str(k): v for k, v in ext_mapping.items()}
                }
                with open(os.path.join(out_dir, "mapping_applique.json"), "w", encoding="utf-8") as map_out:
                    json.dump(mapping_applied, map_out, indent=2, ensure_ascii=False)
                    
                # Build report JSON
                report = {
                    "harmonization_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_local_annotations_files": total_local_files,
                    "total_external_annotations_files": total_ext_files_copied,
                    "annotations_remapped_count": converted_annots,
                    "annotations_unchanged_count": unchanged_annots,
                    "annotations_ignored_count": ignored_annots,
                    "images_isolated_to_manual_verification": manual_files_count,
                    "final_classes": final_classes
                }
                with open(os.path.join(out_dir, "rapport_harmonisation.json"), "w", encoding="utf-8") as rep_out:
                    json.dump(report, rep_out, indent=2, ensure_ascii=False)
                    
                # Clean up manual check folders if empty
                if manual_files_count == 0:
                    try:
                        shutil.rmtree(os.path.join(out_dir, "a_verifier_manuellement"))
                    except Exception:
                        pass
                        
                status.update(label="🎉 Harmonisation finie avec succès !", state="complete")
                st.balloons()
                
                # Show complete report summary
                st.success("🏆 L'harmonisation globale des classes est terminée ! Les données sont prêtes sous `dataset_final_harmonise/`.")
                
                st.markdown("<div style='background-color:#1E293B; border-radius: 8px; padding:15px; margin-top:15px;'>", unsafe_allow_html=True)
                st.markdown("#### 📋 Synthèse statistique de l'harmonisation")
                
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Total d'images harmonisées", f"{total_local_files + total_ext_files_copied - manual_files_count}")
                    st.metric("Annotations remappées", f"{converted_annots}")
                with sc2:
                    st.metric("Images isolées (Erreurs / Ambiguës)", f"{manual_files_count}")
                    st.metric("Annotations inchangées", f"{unchanged_annots}")
                with sc3:
                    st.metric("Poids de backup créés", f"1 sauvegarde")
                    st.metric("Annotations ignorées", f"{ignored_annots}")
                st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as ex_h:
            st.error(f"Une erreur est survenue pendant l'harmonisation : {ex_h}")
            st.info("Assurez-vous que l'application possède les droits de lecture/écriture nécessaires dans le dossier du projet.")
