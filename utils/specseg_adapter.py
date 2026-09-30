"""Adaptateur SpecSeg : second expert de detection des reflets speculaires.

SpecSeg tourne en parallele de YOLO et ne modifie jamais l'image analysee.
Le backend par defaut est le Hugging Face Space ``atifanwerPK/SpecSeg``
(via ``gradio_client``). L'interface ``predict_mask(image_rgb)`` reste
identique quel que soit le backend, ce qui permettra plus tard de basculer
vers un SpecSeg local sans toucher au reste de l'application.

Endpoint reel du Space (decouvert via ``Client.view_api()``)::

    predict(inputimage: filepath, api_name="/predict") -> specularmask: filepath
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from collections import OrderedDict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Backend actif : "huggingface" (distant) aujourd'hui, "local" plus tard.
SPECSEG_BACKEND = "huggingface"

SPECSEG_HF_SPACE = "atifanwerPK/SpecSeg"
SPECSEG_HF_API_NAME = "/predict"

# Timeout (secondes) pour un appel distant, Space endormi inclus.
SPECSEG_TIMEOUT_SECONDS = 120

# Nombre maximum de masques gardes en cache (une entree par image).
SPECSEG_CACHE_MAX_ENTRIES = 32


class SpecSegError(RuntimeError):
    """Erreur non bloquante levee quand SpecSeg est indisponible."""


def image_sha256(image_rgb: np.ndarray) -> str:
    """Hash SHA256 du contenu d'une image NumPy RGB."""
    array = np.ascontiguousarray(image_rgb)
    digest = hashlib.sha256(array.tobytes())
    digest.update(str(array.shape).encode("utf-8"))
    return digest.hexdigest()


def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask.shape == (target_h, target_w):
        return mask
    try:
        import cv2

        return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image

        pil = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
        pil = pil.resize((target_w, target_h), Image.BILINEAR)
        return np.asarray(pil).astype(np.float32) / 255.0


class SpecSegAdapter:
    """Predit un masque de reflets speculaires pour une image RGB.

    ``predict_mask`` retourne toujours un ``np.ndarray`` float32 de shape
    ``(H, W)`` avec des valeurs dans ``[0.0, 1.0]``, aligne exactement sur
    l'image d'entree. Les resultats sont mis en cache par hash SHA256 de
    l'image : une meme frame n'est envoyee qu'une seule fois au backend.
    """

    def __init__(
        self,
        backend: str = SPECSEG_BACKEND,
        timeout_seconds: float = SPECSEG_TIMEOUT_SECONDS,
        cache_max_entries: int = SPECSEG_CACHE_MAX_ENTRIES,
    ) -> None:
        self.backend = backend
        self.timeout_seconds = timeout_seconds
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_max_entries = cache_max_entries
        self._client = None
        self._client_error: Optional[str] = None
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ API

    def predict_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Retourne le masque de reflets, shape (H, W), valeurs [0, 1].

        Leve ``SpecSegError`` si SpecSeg est indisponible (reseau, timeout,
        Space endormi, image invalide...). L'appelant doit traiter cette
        erreur comme non bloquante : YOLO continue normalement.
        """
        image_rgb = self._validate_image(image_rgb)
        key = image_sha256(image_rgb)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        mask = self._predict_uncached(image_rgb)
        mask = self._normalize_mask(mask, image_rgb.shape[0], image_rgb.shape[1])

        with self._lock:
            self._cache[key] = mask
            while len(self._cache) > self._cache_max_entries:
                self._cache.popitem(last=False)
        return mask

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------- internes

    @staticmethod
    def _validate_image(image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb is None:
            raise SpecSegError("Image invalide : None.")
        array = np.asarray(image_rgb)
        if array.ndim != 3 or array.shape[2] != 3:
            raise SpecSegError(
                f"Image invalide : shape {array.shape}, attendu (H, W, 3) RGB."
            )
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise SpecSegError("Image invalide : dimensions nulles.")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    @staticmethod
    def _normalize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.float32)
        if mask.size and mask.max() > 1.0:
            mask = mask / 255.0
        mask = np.clip(mask, 0.0, 1.0)
        mask = _resize_mask(mask, target_h, target_w)
        return np.clip(mask.astype(np.float32), 0.0, 1.0)

    def _predict_uncached(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.backend == "huggingface":
            return self._predict_huggingface(image_rgb)
        if self.backend == "local":
            return self._predict_local(image_rgb)
        raise SpecSegError(f"Backend SpecSeg inconnu : {self.backend!r}.")

    def _get_client(self):
        with self._lock:
            if self._client is not None:
                return self._client
            if self._client_error is not None:
                raise SpecSegError(self._client_error)
        try:
            from gradio_client import Client

            client = Client(SPECSEG_HF_SPACE)
        except Exception as error:  # reseau absent, Space introuvable...
            message = f"Connexion au Space SpecSeg impossible : {error}"
            with self._lock:
                self._client_error = message
            raise SpecSegError(message) from error
        with self._lock:
            self._client = client
        return client

    def _predict_huggingface(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        client = self._get_client()

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="specseg_in_")
            os.close(fd)
            Image.fromarray(image_rgb).save(tmp_path)

            from gradio_client import handle_file

            def call() -> str:
                return client.predict(
                    handle_file(tmp_path), api_name=SPECSEG_HF_API_NAME
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(call)
                try:
                    result_path = future.result(timeout=self.timeout_seconds)
                except FuturesTimeoutError as error:
                    future.cancel()
                    raise SpecSegError(
                        f"Timeout SpecSeg apres {self.timeout_seconds:.0f}s "
                        "(Space endormi ou surcharge)."
                    ) from error

            mask_image = Image.open(result_path).convert("L")
            return np.asarray(mask_image)
        except SpecSegError:
            raise
        except Exception as error:
            raise SpecSegError(f"Echec de la prediction SpecSeg : {error}") from error
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.debug("Fichier temporaire SpecSeg non supprime : %s", tmp_path)

    def _predict_local(self, image_rgb: np.ndarray) -> np.ndarray:
        # Deuxieme etape prevue : SpecSeg local dans un environnement separe
        # (sans TensorFlow dans .venv-gpu). L'interface predict_mask restera
        # identique.
        raise SpecSegError(
            "Backend SpecSeg local non disponible : utilisez 'huggingface'."
        )
