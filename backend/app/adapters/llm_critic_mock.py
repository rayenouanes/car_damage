from __future__ import annotations

from backend.app.models.schemas import LLMCriticDecision, SAM2Mask, VLMTeacherAnalysis


class LLMCriticMock:
    provider = "llm_critic_mock"

    def critique(
        self,
        prediction: dict,
        sam2_mask: SAM2Mask | None,
        vlm_analysis: VLMTeacherAnalysis,
        rag_rules_text: str,
        similar_errors: list[dict],
    ) -> LLMCriticDecision:
        reasons = prediction.get("active_learning_reasons", [])
        if not vlm_analysis.defaut_reel and vlm_analysis.type_erreur_possible:
            decision = "rejeter"
            error_type = vlm_analysis.type_erreur_possible
            final_class = None
        elif not vlm_analysis.bbox_correcte:
            decision = "corriger"
            error_type = "bbox_imprecise"
            final_class = vlm_analysis.classe_probable or prediction["class_name"]
        elif vlm_analysis.a_envoyer_humain or similar_errors:
            decision = "envoyer_humain"
            error_type = "mauvaise_classe" if "classes_confondues" in reasons else None
            final_class = vlm_analysis.classe_probable or prediction["class_name"]
        else:
            decision = "accepter"
            error_type = None
            final_class = prediction["class_name"]
        correction_bbox = "reduire" if "bbox_trop_grande" in reasons else "aucune"
        if sam2_mask is not None and sam2_mask.confidence < 0.45:
            correction_bbox = "deplacer"
        return LLMCriticDecision(
            decision=decision,
            classe_finale=final_class,
            correction_bbox=correction_bbox,
            type_erreur=error_type,
            ajouter_rag=bool(similar_errors) and len(similar_errors) >= 3,
            ajouter_finetuning=decision != "accepter" or vlm_analysis.niveau_doute != "faible",
        )
