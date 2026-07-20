from __future__ import annotations

import io
import json
import os
from collections import Counter

import requests
import streamlit as st
from PIL import Image, ImageDraw

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


API_DEFAULT = os.getenv("AL_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("AL_API_KEY", "")
BACKOFFICE_USERNAME = os.getenv("BACKOFFICE_USERNAME", "admin")
BACKOFFICE_PASSWORD = os.getenv("BACKOFFICE_PASSWORD", "")
CLASSES = ["rayure", "bosse", "impact", "defaut_peinture"]
CLASS_LABELS = {
    "rayure": "Rayure",
    "bosse": "Bosse / dent",
    "impact": "Impact",
    "defaut_peinture": "Defaut peinture / vernis",
}
COLORS = {
    "rayure": "#ef4444", "bosse": "#3b82f6",
    "impact": "#f59e0b", "defaut_peinture": "#8b5cf6",
}
PAGES = [
    "Dashboard jobs",
    "Upload video/images",
    "Analyse video + tracking",
    "Audit / doute",
    "Visualisation frames + bbox",
    "File de review humaine",
    "Correction annotation",
    "Error Bank",
    "Memoire RAG",
    "Export dataset YOLO",
]


st.set_page_config(page_title="Active Learning Carrosserie", layout="wide")


def require_backoffice_login() -> None:
    if not BACKOFFICE_PASSWORD:
        return
    if st.session_state.get("backoffice_authenticated"):
        with st.sidebar:
            st.caption(f"Connecte: {BACKOFFICE_USERNAME}")
            if st.button("Se deconnecter"):
                st.session_state.pop("backoffice_authenticated", None)
                st.rerun()
        return

    st.title("Backoffice Car Damage Detection")
    st.caption("Acces reserve aux personnes autorisees.")
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
st.title("Plateforme Active Learning - Defauts automobiles")
st.caption(
    "Production: YOLO seul. Training: keyframes + SAM2 + VLM + RAG + LLM. "
    "Audit: VLM/LLM seulement en cas de doute."
)

api_url = st.sidebar.text_input("URL FastAPI", API_DEFAULT).rstrip("/")
page = st.sidebar.radio("Navigation", PAGES)


def api(method: str, path: str, **kwargs):
    try:
        headers = dict(kwargs.pop("headers", {}) or {})
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        response = requests.request(
            method, f"{api_url}{path}", timeout=300, headers=headers, **kwargs
        )
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"API {response.status_code}: {detail}")
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return response.content
    except requests.RequestException as exc:
        raise RuntimeError(f"Backend indisponible: {exc}") from exc


try:
    health = api("GET", "/health")
    st.sidebar.success(f"Backend connecte - {health['yolo_provider']}")
    if not health["model_exists"]:
        st.sidebar.info("Mode mock YOLO: aucun poids configure")
except RuntimeError as exc:
    health = None
    st.sidebar.error(str(exc))


def jobs() -> list[dict]:
    if not health:
        return []
    try:
        return api("GET", "/api/jobs")
    except RuntimeError as exc:
        st.error(str(exc))
        return []


def job_selector(key: str, statuses: set[str] | None = None) -> str | None:
    available = jobs()
    if statuses:
        available = [item for item in available if item["status"] in statuses]
    if not available:
        st.info("Aucun job disponible pour cette page.")
        return None
    options = {
        f"{item['created_at'][:19]} | {item['source_name'][:60]} | {item['status']}": item["id"]
        for item in available
    }
    current = st.session_state.get("job_id")
    index = list(options.values()).index(current) if current in options.values() else 0
    label = st.selectbox("Job", list(options), index=index, key=key)
    st.session_state["job_id"] = options[label]
    return options[label]


def annotated_image(image_bytes: bytes, predictions: list[dict]) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["bbox"]
        color = COLORS.get(prediction["class_name"], "white")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, image.width // 400))
        polygon = (prediction.get("sam2") or {}).get("polygon")
        if polygon and len(polygon) >= 3:
            points = [tuple(point) for point in polygon]
            draw.line(points + [points[0]], fill="#22c55e", width=max(2, image.width // 500))
        track = f" | {prediction['track_id']}" if prediction.get("track_id") else ""
        label = f"{CLASS_LABELS.get(prediction['class_name'], prediction['class_name'])} {prediction['confidence']:.2f}{track}"
        draw.text((x1 + 3, max(0, y1 - 14)), label, fill=color)
    return image


def save_correction(prediction_id: str, payload: dict) -> None:
    try:
        api("POST", f"/api/predictions/{prediction_id}/corrections", json=payload)
        st.success("Correction enregistree dans l'Error Bank")
        st.rerun()
    except RuntimeError as exc:
        st.error(str(exc))


if page == "Dashboard jobs":
    st.header("Dashboard jobs")
    all_jobs = jobs()
    total_images = sum(item["image_count"] for item in all_jobs)
    total_predictions = sum(item["prediction_count"] for item in all_jobs)
    total_reviews = sum(item["reviewed_count"] for item in all_jobs)
    cols = st.columns(4)
    cols[0].metric("Jobs", len(all_jobs))
    cols[1].metric("Images / frames", total_images)
    cols[2].metric("Predictions", total_predictions)
    cols[3].metric("Corrections", total_reviews)
    st.metric("Keyframes selectionnees", sum(item.get("keyframe_count", 0) for item in all_jobs))
    if all_jobs:
        st.dataframe(all_jobs, use_container_width=True, hide_index=True)

elif page == "Upload video/images":
    st.header("Upload video/images")
    if health:
        sam2_provider = health.get("sam2_provider", "disabled")
        if sam2_provider == "sam2":
            st.success("SAM2 reel actif")
        elif sam2_provider == "disabled":
            st.warning("SAM2 desactive")
        else:
            st.info("SAM2 fonctionne actuellement en mode mock")
    
    uploads = st.file_uploader(
        "Video MP4/MOV/AVI/MKV/WEBM ou plusieurs images",
        type=["mp4", "mov", "avi", "mkv", "webm", "jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
    mode = st.radio("Echantillonnage video", ["Toutes les X frames", "Toutes les X secondes"])
    every_n_frames = st.number_input("X frames", 1, 10000, 30, disabled=mode != "Toutes les X frames")
    every_seconds = st.number_input(
        "X secondes", 0.1, 120.0, 1.0, 0.1, disabled=mode != "Toutes les X secondes"
    )
    if st.button("Uploader", type="primary", disabled=not uploads or not health):
        file_payload = [("files", (item.name, item.getvalue(), item.type)) for item in uploads]
        form = {"every_n_frames": every_n_frames}
        if mode == "Toutes les X secondes":
            form["every_seconds"] = every_seconds
        try:
            with st.spinner("Stockage et extraction des frames..."):
                result = api("POST", "/api/jobs/upload", files=file_payload, data=form)
            st.session_state["job_id"] = result["job_id"]
            st.success(f"{result['images_created']} image(s)/frame(s) creees")
            for warning in result["warnings"]:
                st.warning(warning)
        except RuntimeError as exc:
            st.error(str(exc))
    job_id = job_selector("upload_job")
    if job_id:
        force = st.checkbox("Recalculer les predictions existantes")
        enable_sam2 = st.checkbox("Activer SAM2 pour les images", value=True)
        if st.button("Lancer YOLO + active learning", type="primary"):
            try:
                with st.spinner("Inference et analyse des cas prioritaires..."):
                    result = api(
                        "POST", f"/api/jobs/{job_id}/infer",
                        json={"force": force, "mode": "training", "enable_sam2": enable_sam2},
                    )
                count = sum(len(image["predictions"]) for image in result["images"])
                st.success(f"Inference terminee: {count} detection(s)")
                st.json(result.get("frame_selection", {}), expanded=False)
            except RuntimeError as exc:
                st.error(str(exc))

elif page == "Analyse video + tracking":
    st.header("Analyse video + tracking")
    if health:
        sam2_provider = health.get("sam2_provider", "disabled")
        if sam2_provider == "sam2":
            st.success("SAM2 reel actif pour les meilleures frames")
        elif sam2_provider == "disabled":
            st.warning("SAM2 desactive : les bbox YOLO restent disponibles")
        else:
            st.info("SAM2 fonctionne actuellement en mode mock")

    video_upload = st.file_uploader(
        "Video a analyser",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        accept_multiple_files=False,
        key="tracking_video",
    )
    sampling_mode = st.radio(
        "Echantillonnage",
        ["Toutes les X secondes", "Toutes les X frames"],
        horizontal=True,
        key="tracking_sampling_mode",
    )
    if sampling_mode == "Toutes les X secondes":
        tracking_seconds = st.number_input(
            "Intervalle en secondes", 0.1, 30.0, 0.5, 0.1, key="tracking_seconds"
        )
        tracking_frames = 30
    else:
        tracking_frames = st.number_input(
            "Intervalle en frames", 1, 1000, 15, key="tracking_frames"
        )
        tracking_seconds = None

    if video_upload is not None:
        st.video(video_upload.getvalue())
    if st.button(
        "Uploader la video",
        type="primary",
        disabled=video_upload is None or not health,
    ):
        form = {"every_n_frames": int(tracking_frames)}
        if tracking_seconds is not None:
            form["every_seconds"] = float(tracking_seconds)
        try:
            with st.spinner("Extraction des frames..."):
                result = api(
                    "POST",
                    "/api/jobs/upload",
                    files=[(
                        "files",
                        (video_upload.name, video_upload.getvalue(), video_upload.type),
                    )],
                    data=form,
                )
            st.session_state["job_id"] = result["job_id"]
            st.success(f"{result['images_created']} frames extraites")
            for warning in result["warnings"]:
                st.warning(warning)
        except RuntimeError as exc:
            st.error(str(exc))

    job_id = job_selector("tracking_job")
    if job_id:
        try:
            selected_job = api("GET", f"/api/jobs/{job_id}")
            video_frames = [
                image for image in selected_job["images"]
                if image.get("frame_index") is not None
            ]
            if not video_frames:
                st.warning("Ce job ne contient pas de frames video")
            else:
                force_tracking = st.checkbox(
                    "Recalculer le tracking existant", key="force_video_tracking"
                )
                if st.button("Lancer YOLO + tracking + SAM2", type="primary"):
                    with st.spinner("Tracking des defauts et segmentation des meilleures frames..."):
                        selected_job = api(
                            "POST",
                            f"/api/jobs/{job_id}/infer",
                            json={"force": force_tracking, "mode": "training"},
                        )
                    summary = selected_job.get("frame_selection", {})
                    st.success(
                        f"{summary.get('tracks', 0)} piste(s), "
                        f"{summary.get('best_frames', 0)} meilleure(s) frame(s)"
                    )

                selected_job = api("GET", f"/api/jobs/{job_id}")
                entries = [
                    (image, prediction)
                    for image in selected_job["images"]
                    for prediction in image["predictions"]
                    if prediction.get("track_id")
                ]
                metrics = st.columns(4)
                metrics[0].metric("Frames extraites", len(video_frames))
                metrics[1].metric("Pistes", len({item[1]["track_id"] for item in entries}))
                metrics[2].metric(
                    "Meilleures frames", len({item[0]["id"] for item in entries})
                )
                metrics[3].metric(
                    "Masques SAM2", sum(item[1].get("sam2") is not None for item in entries)
                )

                if entries:
                    options = {
                        (
                            f"{prediction['track_id']} | "
                            f"{CLASS_LABELS.get(prediction['class_name'], prediction['class_name'])} | "
                            f"t={float(image.get('timestamp_seconds') or 0):.2f}s"
                        ): (image, prediction)
                        for image, prediction in entries
                    }
                    selected_label = st.selectbox("Piste / meilleure frame", list(options))
                    selected_image, selected_prediction = options[selected_label]
                    content = api("GET", f"/api/images/{selected_image['id']}/content")
                    st.image(
                        annotated_image(content, selected_image["predictions"]),
                        use_container_width=True,
                    )
                    st.json(
                        {
                            "track_id": selected_prediction["track_id"],
                            "classe": selected_prediction["class_name"],
                            "frames_suivies": selected_prediction.get("track_length"),
                            "frame_index": selected_image.get("frame_index"),
                            "timestamp_seconds": selected_image.get("timestamp_seconds"),
                            "score_yolo": selected_prediction["confidence"],
                            "score_meilleure_frame": selected_prediction.get("track_score"),
                            "sam2": selected_prediction.get("sam2"),
                        },
                        expanded=False,
                    )
                elif selected_job["status"] == "uploaded":
                    st.info("Lancez YOLO + tracking + SAM2 pour obtenir les meilleures frames")
                else:
                    st.info("Aucune piste de defaut detectee dans cette video")
        except RuntimeError as exc:
            st.error(str(exc))

elif page == "Audit / doute":
    st.header("Mode optionnel audit / doute")
    st.caption("YOLO tourne toujours. VLM/LLM ne sont appeles que pour une detection faible ou incoherente.")
    audit_file = st.file_uploader(
        "Image a auditer", type=["jpg", "jpeg", "png", "bmp", "webp"], key="audit_file"
    )
    audit_threshold = st.slider("Seuil de doute", 0.0, 1.0, 0.70, 0.05)
    if st.button("Lancer l'audit", disabled=not audit_file):
        try:
            result = api(
                "POST", "/api/audit/infer",
                files={"file": (audit_file.name, audit_file.getvalue(), audit_file.type)},
                data={"confidence_threshold": audit_threshold},
            )
            st.image(audit_file.getvalue(), use_container_width=True)
            st.json(result)
        except RuntimeError as exc:
            st.error(str(exc))

elif page == "Visualisation frames + bbox":
    st.header("Visualisation frames + bbox")
    job_id = job_selector("visual_job")
    if job_id:
        try:
            job = api("GET", f"/api/jobs/{job_id}")
            items = [image for image in job["images"] if image["predictions"]]
            if not items:
                st.info("Aucune prediction a afficher. Lancez l'inference.")
            else:
                labels = {
                    f"{image['original_name']} | frame {image['frame_index']} | {image['id'][:8]}": image
                    for image in items
                }
                selected = labels[st.selectbox("Image / frame", list(labels))]
                content = api("GET", f"/api/images/{selected['id']}/content")
                st.image(
                    annotated_image(content, selected["predictions"]), use_container_width=True
                )
                st.json({"image_id": selected["id"], "detections": selected["predictions"]})
        except RuntimeError as exc:
            st.error(str(exc))

elif page == "File de review humaine":
    st.header("File de review humaine")
    job_id = job_selector("review_job", {"ready_for_review"})
    if job_id:
        try:
            queue = api("GET", f"/api/review-queue?job_id={job_id}&limit=500")
            if not queue:
                st.success("La file de review est vide.")
            else:
                prediction = queue[0]
                st.progress(1 / len(queue), text=f"{len(queue)} prediction(s) a revoir")
                left, right = st.columns([3, 2])
                with left:
                    content = api("GET", f"/api/images/{prediction['image_id']}/content")
                    st.image(annotated_image(content, [prediction]), use_container_width=True)
                with right:
                    st.subheader("Prediction YOLO")
                    st.json(
                        {
                            "classe": prediction["class_name"],
                            "confiance": prediction["confidence"],
                            "bbox": prediction["bbox"],
                            "priorites": prediction["active_learning_reasons"],
                        }
                    )
                    st.subheader("Analyse VLM")
                    st.json(prediction.get("vlm") or {"info": "Cas non prioritaire"})
                    st.subheader("Masque SAM2")
                    st.json(prediction.get("sam2") or {"info": "SAM2 desactive ou indisponible"})
                    st.subheader("Decision LLM")
                    st.json(prediction.get("llm") or {"info": "Cas non prioritaire"})

                st.subheader("Correction humaine")
                button_cols = st.columns(4)
                actions = [
                    ("Accepter", "accept"), ("Rejeter", "reject"),
                    ("Marquer comme reflet", "reflection"), ("Marquer comme salete", "dirt"),
                    ("Marquer comme ombre", "shadow"),
                    ("Envoyer dans Error Bank", "error_bank"),
                ]
                for index, (label, action) in enumerate(actions):
                    if button_cols[index % 4].button(label, key=f"{action}_{prediction['id']}"):
                        save_correction(
                            prediction["id"],
                            {"action": action, "masque_valide": action == "accept"},
                        )
                if st.button("Valider masque SAM2", disabled=prediction.get("sam2") is None):
                    save_correction(
                        prediction["id"], {"action": "accept", "masque_valide": True}
                    )
                class_choice = st.selectbox("Nouvelle classe", CLASSES, format_func=CLASS_LABELS.get)
                if st.button("Changer classe"):
                    save_correction(
                        prediction["id"],
                        {"action": "change_class", "classe_finale": class_choice},
                    )
                with st.expander("Ajouter regle RAG depuis ce cas"):
                    with st.form("review_rag_rule"):
                        title = st.text_input("Titre")
                        text = st.text_area("Nouvelle regle")
                        add_rule = st.form_submit_button("Ajouter regle RAG")
                    if add_rule:
                        api(
                            "POST", "/api/rag/rules",
                            json={"title": title, "text": text, "tags": [prediction["class_name"]]},
                        )
                        st.success("Regle ajoutee")
        except RuntimeError as exc:
            st.error(str(exc))

elif page == "Correction annotation":
    st.header("Correction annotation")
    job_id = job_selector("correction_job", {"ready_for_review"})
    if job_id:
        try:
            queue = api("GET", f"/api/review-queue?job_id={job_id}&limit=500")
            if not queue:
                st.info("Aucune annotation en attente.")
            else:
                options = {
                    f"{item['original_name']} | {item['class_name']} {item['confidence']:.2f} | {item['id'][:8]}": item
                    for item in queue
                }
                selected = options[st.selectbox("Prediction", list(options))]
                content = api("GET", f"/api/images/{selected['image_id']}/content")
                left, right = st.columns([3, 2])
                left.image(annotated_image(content, [selected]), use_container_width=True)
                with right:
                    with st.form("manual_correction"):
                        class_name = st.selectbox(
                            "Classe finale", CLASSES,
                            index=CLASSES.index(selected["class_name"]), format_func=CLASS_LABELS.get,
                        )
                        bbox = selected["bbox"]
                        x1 = st.number_input("x1", value=float(bbox[0]), min_value=0.0)
                        y1 = st.number_input("y1", value=float(bbox[1]), min_value=0.0)
                        x2 = st.number_input("x2", value=float(bbox[2]), min_value=0.0)
                        y2 = st.number_input("y2", value=float(bbox[3]), min_value=0.0)
                        note = st.text_area("Note")
                        default_mask = (selected.get("sam2") or {}).get("polygon") or []
                        mask_text = st.text_area(
                            "Polygone masque [[x,y], ...]",
                            value=json.dumps(default_mask),
                        )
                        mask_valid = st.checkbox("Masque valide", value=bool(default_mask))
                        submitted = st.form_submit_button("Enregistrer classe + bbox")
                if submitted:
                    try:
                        parsed_mask = json.loads(mask_text) if mask_text.strip() else None
                        save_correction(
                            selected["id"],
                            {
                                "action": "change_class",
                                "classe_finale": class_name,
                                "bbox_finale": [x1, y1, x2, y2],
                                "masque_final": parsed_mask,
                                "masque_valide": mask_valid,
                                "type_erreur": "bbox_imprecise" if [x1, y1, x2, y2] != bbox else "mauvaise_classe",
                                "note": note,
                            },
                        )
                    except json.JSONDecodeError:
                        st.error("Le polygone du masque n'est pas un JSON valide.")
        except RuntimeError as exc:
            st.error(str(exc))

elif page == "Error Bank":
    st.header("Error Bank")
    try:
        records = api("GET", "/api/error-bank?limit=1000") if health else []
        type_counts = Counter(record.get("type_erreur") or "validation" for record in records)
        st.write(dict(type_counts))
        st.metric("Corrections historisees", len(records))
        for record in records:
            payload = record.get("record", {})
            title = (
                f"{record['created_at'][:19]} | {record.get('type_erreur') or 'validation'} | "
                f"{payload.get('image_path', '')}"
            )
            with st.expander(title):
                st.json(payload)
    except RuntimeError as exc:
        st.error(str(exc))

elif page == "Memoire RAG":
    st.header("Memoire RAG")
    left, right = st.columns(2)
    with left:
        with st.form("add_rule"):
            title = st.text_input("Titre")
            text = st.text_area("Regle metier")
            tags = st.text_input("Tags, separes par des virgules")
            submitted = st.form_submit_button("Ajouter une regle")
        if submitted:
            try:
                api(
                    "POST", "/api/rag/rules",
                    json={
                        "title": title, "text": text,
                        "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                    },
                )
                st.success("Regle ajoutee")
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
        query = st.text_input("Rechercher des regles")
        if query:
            try:
                result = api("POST", "/api/rag/search", json={"query": query, "limit": 5})
                st.text(result["text"])
            except RuntimeError as exc:
                st.error(str(exc))
    with right:
        try:
            for rule in api("GET", "/api/rag/rules") if health else []:
                with st.expander(rule["title"]):
                    st.write(rule["text"])
                    st.caption(", ".join(rule.get("tags", [])))
        except RuntimeError as exc:
            st.error(str(exc))

elif page == "Export dataset YOLO":
    st.header("Export dataset YOLO")
    job_id = job_selector("export_job", {"ready_for_review"})
    if job_id:
        train = st.slider("Train", 0.0, 1.0, 0.70, 0.05)
        val = st.slider("Validation", 0.0, 1.0, 0.20, 0.05)
        test = round(1.0 - train - val, 2)
        st.metric("Test", f"{test:.0%}")
        valid = test >= 0
        if not valid:
            st.error("Train + validation ne doit pas depasser 100 %.")
        if st.button("Generer dataset YOLO", type="primary", disabled=not valid):
            try:
                summary = api(
                    "POST", f"/api/jobs/{job_id}/export-yolo",
                    json={
                        "train": train, "val": val, "test": test,
                        "annotation_format": "segmentation",
                    },
                )
                st.session_state["export_summary"] = summary
            except RuntimeError as exc:
                st.error(str(exc))
        summary = st.session_state.get("export_summary")
        if summary:
            st.json(summary, expanded=False)
            archive = api("GET", f"/api/exports/{summary['archive_name']}")
            st.download_button(
                "Telecharger le ZIP", archive, summary["archive_name"], "application/zip"
            )
            st.subheader("Reentrainement YOLO segmentation")
            epochs = st.number_input("Epochs", 1, 1000, 50)
            if st.button("Preparer / lancer le reentrainement"):
                try:
                    training = api(
                        "POST", f"/api/exports/{summary['export_id']}/train-yolo",
                        json={"epochs": epochs, "image_size": 640, "batch": 8, "device": "cpu"},
                    )
                    st.json(training)
                except RuntimeError as exc:
                    st.error(str(exc))
