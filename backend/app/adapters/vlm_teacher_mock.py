from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.app.models.schemas import SAM2Mask, VLMTeacherAnalysis


class VLMTeacherMock:
    provider = "vlm_teacher_mock"

    def analyze(
        self,
        full_image_path: Path,
        crop_bgr: np.ndarray,
        neighbor_paths: list[Path],
        prediction: dict,
        sam2_mask: SAM2Mask | None,
        rag_rules_text: str,
    ) -> VLMTeacherAnalysis:
        confidence = float(prediction["confidence"])
        brightness = float(crop_bgr.mean()) if crop_bgr.size else 0.0
        bright_ratio = float((crop_bgr > 225).mean()) if crop_bgr.size else 0.0
        reflection_risk = brightness > 175 or bright_ratio > 0.18
        stable_context = len(neighbor_paths) >= 2
        real = confidence >= 0.55 and not (reflection_risk and confidence < 0.75)
        possible_error = "faux_positif_reflet" if reflection_risk else None
        doubt = "eleve" if confidence < 0.40 else ("moyen" if confidence < 0.70 else "faible")
        bbox_correct = "bbox_trop_grande" not in prediction.get("active_learning_reasons", [])
        if sam2_mask is not None and sam2_mask.confidence < 0.45:
            bbox_correct = False
        if real and stable_context:
            reason = "La trace parait coherente et peut etre comparee aux frames voisines."
        elif reflection_risk:
            reason = "La zone est tres brillante et peut suivre un reflet plutot qu'un defaut."
        elif sam2_mask is not None:
            reason = "Le masque SAM2 resserre la zone suspecte; la forme reste a valider humainement."
        else:
            reason = "Le score et les indices visuels restent ambigus; une revue humaine est utile."
        return VLMTeacherAnalysis(
            defaut_reel=real,
            classe_probable=prediction["class_name"] if real else None,
            bbox_correcte=bbox_correct,
            niveau_doute=doubt,
            type_erreur_possible=possible_error,
            a_envoyer_humain=doubt != "faible" or reflection_risk,
            reason_short=reason,
        )
