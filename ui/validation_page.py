"""Page ✅ Validation : revue des validations humaines par bounding box."""

from __future__ import annotations

import streamlit as st

from utils.validation_utils import (
    VALIDATION_CSV_FILENAME,
    export_validation_dataset_csv,
    load_validations,
    validation_summary,
)


def render_validation_page(jsonl_path: str, csv_path: str) -> None:
    st.markdown("<h2 style='color:#38BDF8;'>✅ Validation par bounding box</h2>", unsafe_allow_html=True)
    st.caption(
        "Chaque détection est validée individuellement (TP / FP / Unsure) depuis la page "
        "🔍 Analyse, juste sous chaque boîte. Cette page permet de revoir et d'exporter "
        "les validations enregistrées — la vérité terrain de l'Active Learning."
    )

    records = load_validations(jsonl_path)
    if not records:
        st.info("Aucune validation enregistrée pour le moment. Validez des détections depuis la page 🔍 Analyse.")
        return

    summary = validation_summary(records)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ True Positives", summary["TP"])
    col2.metric("❌ False Positives", summary["FP"])
    col3.metric("❔ Unsure", summary["UNSURE"])
    col4.metric("🖼️ Images validées", summary["images"])

    st.markdown("#### 📋 Validations enregistrées")
    classes = sorted({r.get("class_name", "") for r in records if r.get("class_name")})
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        class_filter = st.multiselect("Classe", classes, key="validation_class_filter")
    with f_col2:
        label_filter = st.multiselect(
            "Validation", ["TP", "FP", "UNSURE"], key="validation_label_filter"
        )

    filtered = [
        r for r in records
        if (not class_filter or r.get("class_name") in class_filter)
        and (not label_filter or r.get("validation") in label_filter)
    ]
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
    st.caption(f"{len(filtered)} validation(s) affichée(s) sur {len(records)}.")

    if st.button("📄 Exporter le dataset de validation (CSV)", key="validation_export_csv"):
        try:
            exported = export_validation_dataset_csv(jsonl_path, csv_path)
            st.success(f"{exported} ligne(s) TP/FP exportée(s) vers `{VALIDATION_CSV_FILENAME}`.")
        except Exception as export_error:
            st.error(f"Export CSV impossible : {export_error}")
