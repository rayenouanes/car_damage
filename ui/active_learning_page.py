"""Page 🧠 Active Learning : exploitation des validations humaines.

Statistiques TP / FP / Unsure, filtres (classe, confiance, validation, date)
et hard negative mining : les faux positifs validés sont listés pour être
corrigés puis intégrés au prochain entraînement.
"""

from __future__ import annotations

import streamlit as st

from utils.validation_utils import load_validations, validation_summary


def render_active_learning_page(jsonl_path: str) -> None:
    st.markdown("<h2 style='color:#38BDF8;'>🧠 Active Learning</h2>", unsafe_allow_html=True)
    st.caption(
        "Workflow : YOLO détecte → validation humaine par bounding box → les faux positifs "
        "deviennent des hard negatives → annotations corrigées → prochain entraînement."
    )

    records = load_validations(jsonl_path)
    if not records:
        st.info(
            "Aucune validation disponible. Validez des détections (TP / FP / Unsure) "
            "depuis la page 🔍 Analyse pour alimenter l'Active Learning."
        )
        return

    summary = validation_summary(records)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ TP", summary["TP"])
    col2.metric("❌ FP (hard negatives)", summary["FP"])
    col3.metric("❔ Unsure", summary["UNSURE"])
    col4.metric("🖼️ Images corrigées", summary["images"])

    st.markdown("#### 🔎 Filtres")
    classes = sorted({r.get("class_name", "") for r in records if r.get("class_name")})
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        class_filter = st.multiselect("Classe", classes, key="al_class_filter")
    with f_col2:
        label_filter = st.multiselect(
            "Validation", ["TP", "FP", "UNSURE"], default=["FP"], key="al_label_filter"
        )
    with f_col3:
        max_conf = st.slider(
            "Confiance YOLO maximale", 0.0, 1.0, 1.0, 0.05, key="al_max_conf",
            help="Ex. 0.5 pour récupérer toutes les détections à faible confiance.",
        )

    dates = sorted({(r.get("timestamp", "") or "")[:10] for r in records if r.get("timestamp")})
    date_filter = st.multiselect("Date", dates, key="al_date_filter")

    filtered = [
        r for r in records
        if (not class_filter or r.get("class_name") in class_filter)
        and (not label_filter or r.get("validation") in label_filter)
        and float(r.get("yolo_confidence", 0.0)) <= max_conf
        and (not date_filter or (r.get("timestamp", "") or "")[:10] in date_filter)
    ]

    st.markdown("#### 📋 Détections sélectionnées")
    table_rows = [
        {
            "Date": r.get("timestamp", ""),
            "Image": r.get("image_id", ""),
            "Classe": r.get("class_name", ""),
            "Confiance YOLO": round(float(r.get("yolo_confidence", 0.0)), 3),
            "BBox": r.get("bbox", []),
            "Validation": r.get("validation", ""),
        }
        for r in reversed(filtered)
    ]
    st.dataframe(table_rows, use_container_width=True)
    st.caption(f"{len(filtered)} détection(s) sélectionnée(s) sur {len(records)} validation(s).")

    fp_images = sorted({
        r.get("image_id", "") for r in records
        if r.get("validation") == "FP" and r.get("image_id")
    })
    if fp_images:
        with st.expander(f"🧲 Hard negatives : {len(fp_images)} image(s) avec faux positifs"):
            st.caption(
                "Corrigez ces images dans la page ✏️ Annotation puis relancez un "
                "entraînement : les faux positifs validés servent de hard negatives."
            )
            for image_id in fp_images:
                st.markdown(f"- `{image_id}`")
