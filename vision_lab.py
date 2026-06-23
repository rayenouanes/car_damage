import os
import importlib.util
import json
from datetime import datetime
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


st.set_page_config(
    page_title="Vision Lab - Car Defects",
    page_icon="VL",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PRESETS = {
    "Rayures fines": {
        "white_balance": True,
        "clahe": True,
        "clahe_clip": 2.8,
        "clahe_grid": 8,
        "gamma": 0.95,
        "denoise": 4,
        "glare": True,
        "glare_v": 232,
        "glare_s": 68,
        "glare_strength": 0.42,
        "detail": False,
        "unsharp": True,
        "unsharp_amount": 1.25,
        "unsharp_sigma": 1.1,
        "scratch_kernel": 19,
        "scratch_threshold": 42,
        "canny_low": 35,
        "canny_high": 115,
    },
    "Reflets forts": {
        "white_balance": True,
        "clahe": True,
        "clahe_clip": 2.0,
        "clahe_grid": 8,
        "gamma": 1.05,
        "denoise": 7,
        "glare": True,
        "glare_v": 218,
        "glare_s": 82,
        "glare_strength": 0.62,
        "detail": False,
        "unsharp": True,
        "unsharp_amount": 0.85,
        "unsharp_sigma": 1.4,
        "scratch_kernel": 17,
        "scratch_threshold": 48,
        "canny_low": 45,
        "canny_high": 135,
    },
    "Contraste faible": {
        "white_balance": True,
        "clahe": True,
        "clahe_clip": 3.2,
        "clahe_grid": 8,
        "gamma": 0.72,
        "denoise": 3,
        "glare": False,
        "glare_v": 235,
        "glare_s": 70,
        "glare_strength": 0.4,
        "detail": True,
        "unsharp": True,
        "unsharp_amount": 0.9,
        "unsharp_sigma": 1.0,
        "scratch_kernel": 15,
        "scratch_threshold": 40,
        "canny_low": 30,
        "canny_high": 105,
    },
    "Diagnostic brut": {
        "white_balance": False,
        "clahe": False,
        "clahe_clip": 2.0,
        "clahe_grid": 8,
        "gamma": 1.0,
        "denoise": 0,
        "glare": False,
        "glare_v": 235,
        "glare_s": 70,
        "glare_strength": 0.4,
        "detail": False,
        "unsharp": False,
        "unsharp_amount": 0.0,
        "unsharp_sigma": 1.0,
        "scratch_kernel": 13,
        "scratch_threshold": 45,
        "canny_low": 40,
        "canny_high": 120,
    },
}


REAL_ESRGAN_WEIGHT_CANDIDATES = [
    "RealESRGAN_x4plus.pth",
    "realesrgan_x4plus.pth",
    "RealESRGAN_x2plus.pth",
    os.path.join("weights", "RealESRGAN_x4plus.pth"),
    os.path.join("weights", "RealESRGAN_x2plus.pth"),
    os.path.join("models_history", "RealESRGAN_x4plus.pth"),
]


CONCEPT_ROWS = [
    ("CNN", "Base des modeles de detection type YOLO", "Tres forte"),
    ("Convolution", "Detecte textures, bords, micro-lignes et motifs locaux", "Tres forte"),
    ("Pooling", "Rend les indices plus robustes aux petites variations", "Forte"),
    ("ReLU / activations", "Garde les signaux utiles et evite des calculs trop lents", "Forte"),
    ("Softmax / scores", "Transforme les sorties en scores de classe lisibles", "Forte"),
    ("Cross-Entropy / loss", "Mesure l'erreur de classification pendant l'apprentissage", "Forte"),
    ("Forward -> loss -> backward -> step", "Cycle complet d'entrainement du reseau", "Tres forte"),
    ("Learning rate", "Regle la vitesse d'apprentissage du modele", "Forte"),
    ("Overfitting", "Risque majeur sur petits datasets de rayures/reflets", "Tres forte"),
    ("Matrice de confusion", "Montre les confusions entre rayure, reflet, poussiere, fissure", "Tres forte"),
    ("Sigmoide", "Utile surtout pour presence/absence ou multi-label", "Moyenne"),
    ("Tanh", "Moins prioritaire pour ton pipeline actuel", "Faible"),
    ("Leaky ReLU", "Alternative quand des neurones deviennent inactifs", "Moyenne"),
    ("Vanishing gradient", "A comprendre pour les reseaux profonds", "Moyenne"),
    ("Neurones morts", "Risque avec ReLU si le signal disparait", "Moyenne"),
    ("Perceptron", "Brique historique pour comprendre le neurone artificiel", "Moyenne"),
]


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(to_rgb(image_bgr))


def ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def normalize_u8(image: np.ndarray) -> np.ndarray:
    if image.max() == image.min():
        return np.zeros_like(image, dtype=np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def decode_upload(uploaded_file) -> np.ndarray | None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def resize_max_side(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    longest = max(width, height)
    if longest <= max_side:
        return image_bgr
    scale = max_side / float(longest)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def gray_world_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    work = image_bgr.astype(np.float32)
    channel_means = work.reshape(-1, 3).mean(axis=0)
    target = channel_means.mean()
    scale = target / (channel_means + 1e-6)
    return np.clip(work * scale, 0, 255).astype(np.uint8)


def apply_clahe_lab(image_bgr: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    grid_size = max(2, int(grid_size))
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid_size, grid_size))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_gamma(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.2, float(gamma))
    lookup = ((np.arange(256) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(image_bgr, lookup)


def apply_unsharp(image_bgr: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    if amount <= 0:
        return image_bgr
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), max(0.1, float(sigma)))
    return cv2.addWeighted(image_bgr, 1.0 + float(amount), blurred, -float(amount), 0)


def reduce_glare(
    image_bgr: np.ndarray,
    value_threshold: int,
    saturation_threshold: int,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = ((value >= int(value_threshold)) & (saturation <= int(saturation_threshold))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    softened = image_bgr.copy()
    if mask.any():
        hsv_soft = hsv.copy()
        glare_pixels = mask > 0
        hsv_soft[:, :, 2][glare_pixels] = np.clip(
            hsv_soft[:, :, 2][glare_pixels].astype(np.float32) * (1.0 - float(strength)),
            0,
            255,
        ).astype(np.uint8)
        softened = cv2.cvtColor(hsv_soft, cv2.COLOR_HSV2BGR)
        inpainted = cv2.inpaint(softened, mask, 3, cv2.INPAINT_TELEA)
        softened = cv2.addWeighted(softened, 0.65, inpainted, 0.35, 0)

    return softened, mask


def attenuate_glare_for_model(
    image_bgr: np.ndarray,
    value_threshold: int,
    saturation_threshold: int,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1]
    mask = ((value >= int(value_threshold)) & (saturation <= int(saturation_threshold))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    if not mask.any():
        return image_bgr, mask

    local_value = cv2.GaussianBlur(value, (0, 0), 7)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 5)
    alpha = np.clip(alpha * float(strength), 0, 0.45)
    value = value * (1.0 - alpha) + local_value * alpha
    hsv[:, :, 2] = np.clip(value, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), mask


def line_kernel(length: int, angle: int) -> np.ndarray:
    length = ensure_odd(length, minimum=5)
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    if angle == 0:
        kernel[center, :] = 1
    elif angle == 90:
        kernel[:, center] = 1
    elif angle == 45:
        np.fill_diagonal(np.fliplr(kernel), 1)
    else:
        np.fill_diagonal(kernel, 1)
    return kernel


def build_scratch_map(
    image_bgr: np.ndarray,
    kernel_length: int,
    threshold: int,
    canny_low: int,
    canny_high: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    oriented_maps = []
    for angle in (0, 45, 90, 135):
        kernel = line_kernel(kernel_length, angle)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        oriented_maps.append(cv2.max(blackhat, tophat))

    line_response = oriented_maps[0]
    for response in oriented_maps[1:]:
        line_response = cv2.max(line_response, response)

    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edges = cv2.Canny(gray, int(canny_low), int(canny_high))
    heat = cv2.addWeighted(normalize_u8(line_response), 0.65, normalize_u8(gradient), 0.25, 0)
    heat = cv2.addWeighted(heat, 0.85, edges, 0.15, 0)
    heat = normalize_u8(heat)
    _, binary = cv2.threshold(heat, int(threshold), 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return heat, binary, edges


def apply_pseudo_photometric_stereo(
    image_bgr: np.ndarray,
    blur_size: int = 41,
    normal_strength: float = 2.2,
    light_angle: int = 315,
    light_elevation: int = 45,
    glare_value: int = 225,
    glare_saturation: int = 80,
) -> tuple[np.ndarray, dict]:
    balanced = gray_world_white_balance(image_bgr)
    deglared, glare_mask = reduce_glare(
        balanced,
        value_threshold=glare_value,
        saturation_threshold=glare_saturation,
        strength=0.55,
    )

    lab = cv2.cvtColor(deglared, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32) / 255.0
    blur_size = ensure_odd(blur_size, minimum=9)
    illumination = cv2.GaussianBlur(l_channel, (blur_size, blur_size), 0)
    reflectance = np.log(l_channel + 1e-4) - np.log(illumination + 1e-4)
    reflectance_u8 = normalize_u8(reflectance)

    grad_x = cv2.Sobel(reflectance_u8.astype(np.float32) / 255.0, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(reflectance_u8.astype(np.float32) / 255.0, cv2.CV_32F, 0, 1, ksize=3)
    nx = -grad_x * float(normal_strength)
    ny = -grad_y * float(normal_strength)
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    azimuth = np.deg2rad(float(light_angle))
    elevation = np.deg2rad(float(light_elevation))
    light = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float32,
    )
    shading = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1)
    shading_u8 = normalize_u8(shading)

    surface_l = cv2.addWeighted(reflectance_u8, 0.72, shading_u8, 0.28, 0)
    surface_l = cv2.equalizeHist(surface_l)
    lab[:, :, 0] = cv2.addWeighted(lab[:, :, 0], 0.35, surface_l, 0.65, 0)
    surface_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    surface_bgr = apply_unsharp(surface_bgr, 0.65, 1.0)

    normal_map = np.dstack(
        [
            normalize_u8((nx + 1.0) * 0.5),
            normalize_u8((ny + 1.0) * 0.5),
            normalize_u8((nz + 1.0) * 0.5),
        ]
    )
    return surface_bgr, {
        "glare_mask": glare_mask,
        "reflectance": reflectance_u8,
        "shading": shading_u8,
        "normal_map": normal_map,
    }


def apply_scratch_visibility_filter(image_bgr: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    work = gray_world_white_balance(image_bgr)
    work, glare_mask = reduce_glare(
        work,
        cfg.get("glare_v", 230),
        cfg.get("glare_s", 75),
        max(0.35, cfg.get("glare_strength", 0.45)),
    )
    if cfg.get("denoise", 0) > 0:
        strength = int(cfg["denoise"])
        work = cv2.bilateralFilter(work, d=7, sigmaColor=strength * 8, sigmaSpace=strength * 8)
    work = apply_clahe_lab(work, cfg.get("clahe_clip", 2.6), cfg.get("clahe_grid", 8))
    work = apply_unsharp(work, max(1.0, cfg.get("unsharp_amount", 1.1)), cfg.get("unsharp_sigma", 1.0))
    heat, binary, edges = build_scratch_map(
        work,
        cfg.get("scratch_kernel", 19),
        max(18, cfg.get("scratch_threshold", 42)),
        cfg.get("canny_low", 35),
        cfg.get("canny_high", 115),
    )

    overlay = work.copy()
    overlay[binary > 0] = (0, 255, 255)
    highlighted = cv2.addWeighted(work, 0.78, overlay, 0.22, 0)
    return highlighted, {
        "scratch_heat": heat,
        "scratch_binary": binary,
        "edges": edges,
        "glare_mask": glare_mask,
    }


def find_existing_realesrgan_weight() -> str:
    for candidate in REAL_ESRGAN_WEIGHT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return ""


def opencv_super_resolution(image_bgr: np.ndarray, scale: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    resized = cv2.resize(
        image_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_LANCZOS4,
    )
    resized = cv2.detailEnhance(resized, sigma_s=8, sigma_r=0.10)
    return apply_unsharp(resized, 0.55, 1.1)


@st.cache_resource(show_spinner=False)
def load_realesrganer(weights_path: str, scale: int, tile: int):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )
    return RealESRGANer(
        scale=scale,
        model_path=weights_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )


def apply_realesrgan_or_fallback(
    image_bgr: np.ndarray,
    scale: int,
    weights_path: str,
    tile: int,
    use_real_esrgan: bool,
) -> tuple[np.ndarray, str]:
    scale = int(scale)
    if use_real_esrgan and weights_path and os.path.exists(weights_path):
        try:
            upsampler = load_realesrganer(weights_path, scale, int(tile))
            output, _ = upsampler.enhance(image_bgr, outscale=scale)
            return output, "Real-ESRGAN local"
        except Exception as exc:
            fallback = opencv_super_resolution(image_bgr, scale)
            return fallback, f"Fallback OpenCV SR - Real-ESRGAN indisponible: {exc}"

    fallback = opencv_super_resolution(image_bgr, scale)
    if use_real_esrgan:
        return fallback, "Fallback OpenCV SR - poids Real-ESRGAN introuvables"
    return fallback, "Fallback OpenCV SR"


def process_image(image_bgr: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    work = image_bgr.copy()
    diagnostics = {}

    if cfg["white_balance"]:
        work = gray_world_white_balance(work)

    if cfg["glare"]:
        work, diagnostics["glare_mask"] = reduce_glare(
            work,
            cfg["glare_v"],
            cfg["glare_s"],
            cfg["glare_strength"],
        )
    else:
        diagnostics["glare_mask"] = np.zeros(work.shape[:2], dtype=np.uint8)

    denoise_strength = int(cfg["denoise"])
    if denoise_strength > 0:
        work = cv2.bilateralFilter(work, d=7, sigmaColor=denoise_strength * 10, sigmaSpace=denoise_strength * 10)

    if cfg["clahe"]:
        work = apply_clahe_lab(work, cfg["clahe_clip"], cfg["clahe_grid"])

    if abs(float(cfg["gamma"]) - 1.0) > 0.01:
        work = apply_gamma(work, cfg["gamma"])

    if cfg["detail"]:
        work = cv2.detailEnhance(work, sigma_s=10, sigma_r=0.12)

    if cfg["unsharp"]:
        work = apply_unsharp(work, cfg["unsharp_amount"], cfg["unsharp_sigma"])

    scratch_heat, scratch_binary, edges = build_scratch_map(
        work,
        cfg["scratch_kernel"],
        cfg["scratch_threshold"],
        cfg["canny_low"],
        cfg["canny_high"],
    )
    diagnostics["scratch_heat"] = scratch_heat
    diagnostics["scratch_binary"] = scratch_binary
    diagnostics["edges"] = edges
    return work, diagnostics


def apply_yolo_safe_enhancement(image_bgr: np.ndarray, cfg: dict, profile: dict | None = None) -> tuple[np.ndarray, dict]:
    profile = profile or image_profile(image_bgr)
    work = image_bgr.copy()
    diagnostics = {}

    balanced = gray_world_white_balance(work)
    work = cv2.addWeighted(work, 0.82, balanced, 0.18, 0)

    glare_strength = 0.22
    if profile.get("glare_ratio", 0) > 2.0:
        glare_strength = 0.34
    elif profile.get("glare_ratio", 0) > 0.5:
        glare_strength = 0.27

    work, diagnostics["glare_mask"] = attenuate_glare_for_model(
        work,
        cfg.get("glare_v", 228),
        cfg.get("glare_s", 82),
        glare_strength,
    )

    if profile.get("noise", 0) > 7.0:
        work = cv2.bilateralFilter(work, d=5, sigmaColor=22, sigmaSpace=22)
    elif profile.get("noise", 0) > 4.5:
        work = cv2.bilateralFilter(work, d=5, sigmaColor=14, sigmaSpace=14)

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clip_limit = 1.45 if profile.get("contrast", 0) < 42 else 1.18
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_safe = clahe.apply(l_channel)
    lab_safe = cv2.merge((l_safe, a_channel, b_channel))
    clahe_bgr = cv2.cvtColor(lab_safe, cv2.COLOR_LAB2BGR)
    work = cv2.addWeighted(work, 0.72, clahe_bgr, 0.28, 0)

    if profile.get("brightness", 128) < 95:
        work = apply_gamma(work, 0.88)
    elif profile.get("brightness", 128) > 175:
        work = apply_gamma(work, 1.08)

    if profile.get("sharpness", 0) < 120:
        sharpened = apply_unsharp(work, 0.38, 1.2)
        work = cv2.addWeighted(work, 0.68, sharpened, 0.32, 0)

    heat, binary, edges = build_scratch_map(
        work,
        cfg.get("scratch_kernel", 17),
        cfg.get("scratch_threshold", 42),
        cfg.get("canny_low", 35),
        cfg.get("canny_high", 120),
    )
    diagnostics["scratch_heat"] = heat
    diagnostics["scratch_binary"] = binary
    diagnostics["edges"] = edges
    return work, diagnostics


def image_metrics(image_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    glare_mask = (hsv[:, :, 2] >= 235) & (hsv[:, :, 1] <= 70)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = hist / max(hist.sum(), 1.0)
    entropy = -float(np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])))
    return {
        "Resolution": f"{image_bgr.shape[1]} x {image_bgr.shape[0]}",
        "Contraste": f"{float(gray.std()):.1f}",
        "Nettete": f"{float(cv2.Laplacian(gray, cv2.CV_64F).var()):.1f}",
        "Pixels reflet": f"{float(glare_mask.mean() * 100):.2f}%",
        "Entropie": f"{entropy:.2f}",
    }


def estimate_noise(gray: np.ndarray) -> float:
    median = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - median.astype(np.float32)
    return float(np.std(residual))


def image_profile(image_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    glare_mask = (value >= 235) & (saturation <= 75)
    highlight_mask = value >= 245
    scratch_heat, _, _ = build_scratch_map(image_bgr, 17, 45, 35, 115)
    return {
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "noise": estimate_noise(gray),
        "glare_ratio": float(glare_mask.mean() * 100.0),
        "highlight_ratio": float(highlight_mask.mean() * 100.0),
        "scratch_response": float(np.percentile(scratch_heat, 92)),
    }


def clamp_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def auto_tune_parameters(image_bgr: np.ndarray, current_cfg: dict, requested_sr_scale: int, realesrgan_weights_path: str) -> tuple[dict, dict, dict, list[str]]:
    profile = image_profile(image_bgr)
    cfg = dict(current_cfg)
    reasons = []

    brightness = profile["brightness"]
    contrast = profile["contrast"]
    sharpness = profile["sharpness"]
    noise = profile["noise"]
    glare_ratio = profile["glare_ratio"]
    highlight_ratio = profile["highlight_ratio"]
    scratch_response = profile["scratch_response"]
    longest_side = max(profile["width"], profile["height"])

    cfg["white_balance"] = True
    cfg["glare"] = glare_ratio > 0.35 or highlight_ratio > 1.2
    if cfg["glare"]:
        cfg["glare_v"] = 216 if glare_ratio > 4.0 else 224 if glare_ratio > 1.2 else 232
        cfg["glare_s"] = 92 if glare_ratio > 4.0 else 78
        cfg["glare_strength"] = 0.68 if glare_ratio > 4.0 else 0.52 if glare_ratio > 1.2 else 0.38
        reasons.append("reflets detectes")
    else:
        cfg["glare_v"] = 235
        cfg["glare_s"] = 70
        cfg["glare_strength"] = 0.35

    cfg["clahe"] = True
    if contrast < 38:
        cfg["clahe_clip"] = 3.6
        reasons.append("contraste faible")
    elif contrast > 72:
        cfg["clahe_clip"] = 1.8
    else:
        cfg["clahe_clip"] = 2.6
    cfg["clahe_grid"] = 8 if longest_side < 1600 else 10

    if brightness < 95:
        cfg["gamma"] = 0.72
        reasons.append("image sombre")
    elif brightness > 175:
        cfg["gamma"] = 1.16
        reasons.append("image tres claire")
    else:
        cfg["gamma"] = 0.95

    if noise > 8.5:
        cfg["denoise"] = 7
        reasons.append("bruit visible")
    elif noise > 5.0:
        cfg["denoise"] = 4
    else:
        cfg["denoise"] = 2

    cfg["detail"] = contrast < 42 and sharpness < 220
    cfg["unsharp"] = True
    if sharpness < 90:
        cfg["unsharp_amount"] = 1.55
        cfg["unsharp_sigma"] = 1.15
        reasons.append("image douce/floue")
    elif sharpness > 520:
        cfg["unsharp_amount"] = 0.65
        cfg["unsharp_sigma"] = 1.35
    else:
        cfg["unsharp_amount"] = 1.1
        cfg["unsharp_sigma"] = 1.0

    auto_kernel = clamp_int(min(profile["width"], profile["height"]) * 0.018, 9, 33)
    cfg["scratch_kernel"] = ensure_odd(auto_kernel, minimum=9)
    if scratch_response < 32:
        cfg["scratch_threshold"] = 30
        reasons.append("rayures peu contrastees")
    elif noise > 8.5:
        cfg["scratch_threshold"] = 52
    else:
        cfg["scratch_threshold"] = 40

    if contrast < 40:
        cfg["canny_low"] = 25
        cfg["canny_high"] = 95
    elif noise > 8.5:
        cfg["canny_low"] = 55
        cfg["canny_high"] = 160
    else:
        cfg["canny_low"] = 35
        cfg["canny_high"] = 120

    if glare_ratio > 2.0:
        pseudo_blur = 61
        pseudo_strength = 2.8
    elif contrast < 40:
        pseudo_blur = 49
        pseudo_strength = 2.6
    else:
        pseudo_blur = 39
        pseudo_strength = 2.1

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    glare_points = np.column_stack(np.where((hsv[:, :, 2] >= cfg["glare_v"]) & (hsv[:, :, 1] <= cfg["glare_s"])))
    if len(glare_points) > 10:
        center_y, center_x = np.array(gray.shape) / 2.0
        mean_y, mean_x = glare_points.mean(axis=0)
        pseudo_light_angle = int((np.degrees(np.arctan2(mean_y - center_y, mean_x - center_x)) + 180) % 360)
    else:
        pseudo_light_angle = 315

    if longest_side < 700:
        sr_scale = 3
        reasons.append("image petite")
    elif longest_side < 1100:
        sr_scale = 2
    else:
        sr_scale = min(int(requested_sr_scale), 2)

    realesrgan_available = bool(importlib.util.find_spec("realesrgan") and importlib.util.find_spec("basicsr"))
    use_real_esrgan_auto = bool(realesrgan_available and realesrgan_weights_path and os.path.exists(realesrgan_weights_path))
    if not reasons:
        reasons.append("image deja assez equilibree")

    filter_params = {
        "pseudo_blur": ensure_odd(pseudo_blur, minimum=15),
        "pseudo_normal_strength": pseudo_strength,
        "pseudo_light_angle": pseudo_light_angle,
        "sr_scale": sr_scale,
        "use_real_esrgan": use_real_esrgan_auto,
    }
    return cfg, filter_params, profile, reasons


def metric_delta(original: dict, enhanced: dict) -> list[dict]:
    rows = []
    for key in ("Contraste", "Nettete", "Pixels reflet", "Entropie"):
        try:
            before = float(str(original[key]).replace("%", ""))
            after = float(str(enhanced[key]).replace("%", ""))
            diff = after - before
            rows.append({"Mesure": key, "Avant": original[key], "Apres": enhanced[key], "Delta": f"{diff:+.2f}"})
        except Exception:
            rows.append({"Mesure": key, "Avant": original.get(key), "Apres": enhanced.get(key), "Delta": "-"})
    return rows


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        return b""
    return buffer.tobytes()


def slugify_label(value: str) -> str:
    value = normalize_prompt_text(value)
    cleaned = []
    for ch in value:
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {" ", "-", "_", "/"}:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def make_json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items() if k not in {"crop"}}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    return value


def save_brain_training_samples(
    reasoned_items: list[dict],
    source_filename: str,
    brain_prompt: str,
    brain_rules: dict,
    include_rejected: bool,
) -> dict:
    root = Path("brain_training_dataset")
    crops_root = root / "crops"
    metadata_root = root / "metadata"
    crops_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_stem = slugify_label(Path(source_filename or "uploaded_image").stem)
    saved_records = []

    for idx, item in enumerate(reasoned_items, start=1):
        if not include_rejected and item.get("reasoning_decision") == "rejeter":
            continue
        final_label = slugify_label(item.get("final_label", "unknown_damage"))
        class_dir = crops_root / final_label
        class_dir.mkdir(parents=True, exist_ok=True)

        sample_id = f"{timestamp}_{source_stem}_{idx:03d}"
        crop_path = class_dir / f"{sample_id}.png"
        metadata_path = metadata_root / f"{sample_id}.json"
        crop = item.get("crop")
        if isinstance(crop, np.ndarray) and crop.size:
            cv2.imwrite(str(crop_path), crop)
        else:
            continue

        record = {
            "sample_id": sample_id,
            "source_filename": source_filename,
            "crop_path": str(crop_path),
            "metadata_path": str(metadata_path),
            "final_label": item.get("final_label"),
            "yolo_label": item.get("label"),
            "yolo_conf": item.get("conf"),
            "decision": item.get("reasoning_decision"),
            "risk": item.get("reasoning_risk"),
            "reasoning": item.get("reasoning"),
            "vehicle_zone": item.get("vehicle_zone"),
            "box": item.get("box"),
            "crop_box": item.get("crop_box"),
            "brain_mode": item.get("brain_mode"),
            "brain_prompt": brain_prompt,
            "brain_rules": brain_rules,
            "metrics": {
                "evidence": item.get("evidence"),
                "glare_cover": item.get("glare_cover"),
                "scratch_p90": item.get("scratch_p90"),
                "edge_density": item.get("edge_density"),
                "geometry_score": item.get("geometry_score"),
                "crop_contrast": item.get("crop_contrast"),
                "crop_sharpness": item.get("crop_sharpness"),
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(record), f, ensure_ascii=False, indent=2)
        saved_records.append(record)

    manifest_path = root / "manifest.jsonl"
    if saved_records:
        with open(manifest_path, "a", encoding="utf-8") as f:
            for record in saved_records:
                f.write(json.dumps(make_json_safe(record), ensure_ascii=False) + "\n")

    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "saved_count": len(saved_records),
        "labels": sorted({str(record["final_label"]) for record in saved_records}),
    }


def summarize_brain_dataset() -> dict:
    root = Path("brain_training_dataset")
    crops_root = root / "crops"
    if not crops_root.exists():
        return {"exists": False, "total": 0, "labels": []}
    label_rows = []
    total = 0
    for label_dir in sorted([path for path in crops_root.iterdir() if path.is_dir()]):
        count = len(list(label_dir.glob("*.png")))
        total += count
        label_rows.append({"Classe": label_dir.name, "Exemples": count})
    return {
        "exists": True,
        "total": total,
        "labels": label_rows,
        "root": str(root),
        "manifest_path": str(root / "manifest.jsonl"),
    }


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str, mtime: float):
    if YOLO is None:
        raise RuntimeError("Le package ultralytics n'est pas disponible.")
    return YOLO(model_path)


def extract_detections(result, source_name: str) -> list[dict]:
    detections = []
    names = getattr(result, "names", {}) or {}
    for idx, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append(
            {
                "id": idx + 1,
                "class_id": cls_id,
                "label": str(names.get(cls_id, f"class_{cls_id}")),
                "conf": conf,
                "box": [x1, y1, x2, y2],
                "source": source_name,
            }
        )
    return detections


def box_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union else 0.0


def is_scratch_label(label: str) -> bool:
    label = label.lower()
    return "scratch" in label or "rayure" in label


def detection_visual_evidence(det: dict, image_shape: tuple[int, int, int], scratch_heat: np.ndarray, glare_mask: np.ndarray, edges: np.ndarray) -> dict:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = det["box"]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    area = max(1, (x2 - x1) * (y2 - y1))
    area_ratio = area / float(width * height)
    heat_roi = scratch_heat[y1:y2, x1:x2]
    glare_roi = glare_mask[y1:y2, x1:x2]
    edges_roi = edges[y1:y2, x1:x2]

    if heat_roi.size == 0:
        scratch_p90 = 0.0
        scratch_mean = 0.0
        glare_cover = 0.0
        edge_density = 0.0
    else:
        scratch_p90 = float(np.percentile(heat_roi, 90))
        scratch_mean = float(np.mean(heat_roi))
        glare_cover = float(np.mean(glare_roi > 0))
        edge_density = float(np.mean(edges_roi > 0))

    evidence = 0.0
    evidence += min(scratch_p90 / 85.0, 1.0) * 0.45
    evidence += min(edge_density / 0.065, 1.0) * 0.35
    evidence += max(0.0, 1.0 - glare_cover) * 0.20
    return {
        "area_ratio": area_ratio,
        "scratch_p90": scratch_p90,
        "scratch_mean": scratch_mean,
        "glare_cover": glare_cover,
        "edge_density": edge_density,
        "evidence": evidence,
    }


def validate_detections(detections: list[dict], image_bgr: np.ndarray, cfg: dict, min_conf: float) -> tuple[list[dict], list[dict], dict]:
    scratch_heat, scratch_binary, edges = build_scratch_map(
        image_bgr,
        cfg.get("scratch_kernel", 17),
        cfg.get("scratch_threshold", 42),
        cfg.get("canny_low", 35),
        cfg.get("canny_high", 120),
    )
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    glare_mask = ((hsv[:, :, 2] >= cfg.get("glare_v", 230)) & (hsv[:, :, 1] <= cfg.get("glare_s", 80))).astype(np.uint8) * 255

    kept = []
    rejected = []
    for det in detections:
        enriched = dict(det)
        evidence = detection_visual_evidence(enriched, image_bgr.shape, scratch_heat, glare_mask, edges)
        enriched.update(evidence)
        reject_reasons = []

        if enriched["conf"] < min_conf:
            reject_reasons.append("score faible")
        if evidence["area_ratio"] < 0.00006:
            reject_reasons.append("boite trop petite")
        if evidence["area_ratio"] > 0.58:
            reject_reasons.append("boite trop grande")
        if evidence["glare_cover"] > 0.50 and enriched["conf"] < 0.72:
            reject_reasons.append("zone reflet")
        if is_scratch_label(enriched["label"]):
            has_line_evidence = evidence["scratch_p90"] >= 34 or evidence["edge_density"] >= 0.018
            if not has_line_evidence and enriched["conf"] < 0.78:
                reject_reasons.append("pas assez de lignes fines")
            if evidence["glare_cover"] > 0.34 and evidence["scratch_p90"] < 48:
                reject_reasons.append("rayure confondue avec reflet")
        elif enriched["conf"] < min_conf + 0.10 and evidence["glare_cover"] > 0.35:
            reject_reasons.append("prediction instable sur reflet")

        if reject_reasons:
            enriched["status"] = "rejetee"
            enriched["decision"] = ", ".join(reject_reasons)
            rejected.append(enriched)
        else:
            enriched["status"] = "gardee"
            enriched["decision"] = "validee"
            kept.append(enriched)

    diagnostics = {
        "scratch_heat": scratch_heat,
        "scratch_binary": scratch_binary,
        "edges": edges,
        "glare_mask": glare_mask,
    }
    return kept, rejected, diagnostics


def merge_duplicate_detections(detections: list[dict], iou_threshold: float = 0.50) -> list[dict]:
    detections = sorted(detections, key=lambda item: item["conf"], reverse=True)
    merged = []
    for det in detections:
        duplicate = False
        for current in merged:
            if current["class_id"] == det["class_id"] and box_iou(current["box"], det["box"]) >= iou_threshold:
                duplicate = True
                if det["conf"] > current["conf"]:
                    current.update(det)
                break
        if not duplicate:
            merged.append(det)
    return merged


def stable_ensemble_detections(original_dets: list[dict], safe_dets: list[dict], source_bgr: np.ndarray, safe_bgr: np.ndarray, cfg: dict, min_conf: float) -> tuple[list[dict], list[dict], dict]:
    original_kept, original_rejected, original_diag = validate_detections(original_dets, source_bgr, cfg, min_conf)
    safe_kept, safe_rejected, safe_diag = validate_detections(safe_dets, safe_bgr, cfg, min_conf)

    final = []
    ensemble_rejected = list(original_rejected) + list(safe_rejected)

    for safe_det in safe_kept:
        matches = [
            orig
            for orig in original_kept
            if orig["class_id"] == safe_det["class_id"] and box_iou(orig["box"], safe_det["box"]) >= 0.22
        ]
        if matches:
            best_match = max(matches, key=lambda item: item["conf"])
            merged = dict(safe_det if safe_det["conf"] >= best_match["conf"] else best_match)
            merged["source"] = "stable original + yolo-safe"
            merged["decision"] = "confirmee par deux images"
            final.append(merged)
        elif is_scratch_label(safe_det["label"]) and safe_det["conf"] >= min_conf + 0.12 and safe_det["evidence"] >= 0.48 and safe_det["glare_cover"] < 0.30:
            promoted = dict(safe_det)
            promoted["source"] = "yolo-safe + evidence rayure"
            promoted["decision"] = "gardee: lignes fines fortes"
            final.append(promoted)
        else:
            rejected = dict(safe_det)
            rejected["status"] = "rejetee"
            rejected["decision"] = "non confirmee par l'original"
            ensemble_rejected.append(rejected)

    for orig_det in original_kept:
        has_match = any(
            det["class_id"] == orig_det["class_id"] and box_iou(det["box"], orig_det["box"]) >= 0.30
            for det in final
        )
        if not has_match and orig_det["conf"] >= min_conf + 0.18:
            promoted = dict(orig_det)
            promoted["source"] = "original haute confiance"
            promoted["decision"] = "gardee depuis image naturelle"
            final.append(promoted)

    final = merge_duplicate_detections(final)
    diagnostics = {
        "original": original_diag,
        "yolo_safe": safe_diag,
        "original_kept": original_kept,
        "safe_kept": safe_kept,
    }
    return final, ensemble_rejected, diagnostics


def generate_tiles(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    tile_size = int(max(256, tile_size))
    step = max(64, int(tile_size * (1.0 - float(overlap))))
    xs = list(range(0, max(1, width - tile_size + 1), step))
    ys = list(range(0, max(1, height - tile_size + 1), step))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))

    tiles = []
    seen = set()
    for y in ys:
        for x in xs:
            x2 = min(width, x + tile_size)
            y2 = min(height, y + tile_size)
            tile = (x, y, x2, y2)
            if tile not in seen:
                seen.add(tile)
                tiles.append(tile)
    return tiles


def prioritize_tiles(image_bgr: np.ndarray, tiles: list[tuple[int, int, int, int]], cfg: dict, max_tiles: int) -> list[tuple[int, int, int, int]]:
    if len(tiles) <= max_tiles:
        return tiles

    scratch_heat, _, edges = build_scratch_map(
        image_bgr,
        cfg.get("scratch_kernel", 17),
        cfg.get("scratch_threshold", 42),
        cfg.get("canny_low", 35),
        cfg.get("canny_high", 120),
    )
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    glare = ((hsv[:, :, 2] >= cfg.get("glare_v", 230)) & (hsv[:, :, 1] <= cfg.get("glare_s", 80))).astype(np.uint8)

    scored = []
    for tile in tiles:
        x1, y1, x2, y2 = tile
        heat_roi = scratch_heat[y1:y2, x1:x2]
        edge_roi = edges[y1:y2, x1:x2]
        glare_roi = glare[y1:y2, x1:x2]
        score = float(np.percentile(heat_roi, 88)) + float(np.mean(edge_roi > 0) * 220.0) - float(np.mean(glare_roi > 0) * 25.0)
        scored.append((score, tile))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [tile for _, tile in scored[:max_tiles]]


def map_zoom_box_to_image(box: list[int], tile: tuple[int, int, int, int], zoom_factor: float) -> list[int]:
    x1, y1, x2, y2 = box
    tile_x1, tile_y1, _, _ = tile
    return [
        int(tile_x1 + x1 / zoom_factor),
        int(tile_y1 + y1 / zoom_factor),
        int(tile_x1 + x2 / zoom_factor),
        int(tile_y1 + y2 / zoom_factor),
    ]


def run_tiled_zoom_detection(
    detector,
    image_bgr: np.ndarray,
    cfg: dict,
    conf: float,
    iou: float,
    tile_size: int,
    overlap: float,
    zoom_factor: float,
    imgsz: int,
    max_tiles: int,
    source_name: str,
) -> tuple[list[dict], list[tuple[int, int, int, int]]]:
    height, width = image_bgr.shape[:2]
    tiles = generate_tiles(width, height, tile_size, overlap)
    selected_tiles = prioritize_tiles(image_bgr, tiles, cfg, max_tiles)
    detections = []

    for tile_index, tile in enumerate(selected_tiles):
        x1, y1, x2, y2 = tile
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if zoom_factor > 1.01:
            crop = cv2.resize(
                crop,
                (int(crop.shape[1] * zoom_factor), int(crop.shape[0] * zoom_factor)),
                interpolation=cv2.INTER_CUBIC,
            )
        result = detector(to_pil(crop), conf=conf, iou=iou, imgsz=int(imgsz), verbose=False)[0]
        tile_dets = extract_detections(result, f"{source_name} tile {tile_index + 1}")
        for det in tile_dets:
            det["box"] = map_zoom_box_to_image(det["box"], tile, zoom_factor)
            det["source"] = f"{source_name} zoom x{zoom_factor:g}"
            det["tile"] = tile_index + 1
            detections.append(det)

    return detections, selected_tiles


def draw_tiles(image_bgr: np.ndarray, tiles: list[tuple[int, int, int, int]]) -> np.ndarray:
    canvas = image_bgr.copy()
    for idx, (x1, y1, x2, y2) in enumerate(tiles):
        color = (255, 190, 80)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, str(idx + 1), (x1 + 6, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return canvas


def draw_detection_list(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    canvas = image_bgr.copy()
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["conf"]
        color = (64, 220, 160) if is_scratch_label(label) else (80, 160, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{idx + 1} {label} {conf:.2f}"
        cv2.putText(canvas, caption, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def detection_rows(detections: list[dict]) -> list[dict]:
    rows = []
    for idx, det in enumerate(detections):
        rows.append(
            {
                "#": idx + 1,
                "Classe": det["label"],
                "Score": f"{det['conf']:.3f}",
                "Source": det.get("source", "-"),
                "Decision": det.get("decision", "-"),
                "Evidence": f"{det.get('evidence', 0):.2f}",
                "Reflet": f"{det.get('glare_cover', 0) * 100:.1f}%",
                "Boite": ",".join(str(v) for v in det["box"]),
            }
        )
    return rows


def crop_with_context(image_bgr: np.ndarray, box: list[int], padding: float = 0.30) -> tuple[np.ndarray, list[int]]:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(width, x2 + pad_x)
    cy2 = min(height, y2 + pad_y)
    return image_bgr[cy1:cy2, cx1:cx2].copy(), [cx1, cy1, cx2, cy2]


def infer_vehicle_zone(box: list[int], image_shape: tuple[int, int, int]) -> str:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)

    if cy < 0.28:
        return "vitres / toit"
    if cy > 0.72:
        if cx < 0.35:
            return "pare-chocs avant bas"
        if cx > 0.72:
            return "roue / aile arriere"
        return "bas de caisse / roue"
    if cx < 0.32:
        return "avant / pare-chocs / calandre"
    if cx < 0.58:
        return "capot / aile avant"
    if cx < 0.82:
        return "portes laterales"
    return "arriere lateral"


def geometry_anomaly_score(crop_bgr: np.ndarray) -> float:
    if crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    crop_area = max(1, crop_bgr.shape[0] * crop_bgr.shape[1])
    contour_area = sum(cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 8)
    edge_density = float(np.mean(edges > 0))
    return float(min(1.0, contour_area / crop_area * 4.0 + edge_density * 2.2))


DEFAULT_BRAIN_PROMPT = """Raisonne comme un expert inspection carrosserie.
Rejette les reflets et les ombres si la zone est brillante mais sans lignes fines.
Confirme une rayure seulement si elle contient une ligne fine ou un bord local coherent.
Si une zone a l'avant ou autour de la plaque est deformee mais ne ressemble pas aux classes YOLO, classe-la front_bumper_damage ou unknown_damage.
Si la prediction YOLO est faible et que la zone ressemble au decor reflete, rejette-la."""


def normalize_prompt_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def parse_reasoning_prompt(prompt: str) -> dict:
    text = normalize_prompt_text(prompt)
    rules = {
        "mode": "equilibre",
        "reject_reflections": True,
        "strict_scratch": True,
        "allow_unknown_damage": True,
        "front_bumper_logic": True,
        "plate_holder_logic": True,
        "door_reflection_skeptic": True,
        "small_scratch_recall": True,
        "glare_reject_threshold": 0.45,
        "scratch_confirm_threshold": 42.0,
        "edge_confirm_threshold": 0.016,
        "geometry_unknown_threshold": 0.56,
        "geometry_reclass_threshold": 0.50,
        "confidence_confirm_threshold": 0.65,
        "unknown_label": "unknown_damage",
        "activated_rules": [],
    }

    if has_any(text, ["conservateur", "strict", "faux positif", "fausse detection", "ne detecte pas n'importe quoi"]):
        rules["mode"] = "conservateur"
        rules["glare_reject_threshold"] = 0.34
        rules["scratch_confirm_threshold"] = 50.0
        rules["edge_confirm_threshold"] = 0.022
        rules["confidence_confirm_threshold"] = 0.72
        rules["activated_rules"].append("Mode conservateur: moins de faux positifs.")

    if has_any(text, ["sensible", "ne rate pas", "rappel", "recall", "detecter plus", "petite rayure", "micro rayure"]):
        rules["mode"] = "sensible"
        rules["scratch_confirm_threshold"] = 34.0
        rules["edge_confirm_threshold"] = 0.012
        rules["confidence_confirm_threshold"] = 0.55
        rules["small_scratch_recall"] = True
        rules["activated_rules"].append("Mode sensible: garde plus facilement les petits defauts.")

    if has_any(text, ["reflet", "reflets", "brillant", "brillance", "ombre", "decor", "batiment reflete"]):
        rules["reject_reflections"] = True
        rules["activated_rules"].append("Reflets: rejeter les zones brillantes sans evidence locale.")

    if has_any(text, ["ne rejette pas les reflets", "garde les reflets", "reflet peut etre defaut"]):
        rules["reject_reflections"] = False
        rules["activated_rules"].append("Reflets: ne pas rejeter automatiquement.")

    if has_any(text, ["rayure fine", "rayures fines", "ligne fine", "lignes fines", "scratch", "micro rayure"]):
        rules["strict_scratch"] = True
        rules["small_scratch_recall"] = True
        rules["activated_rules"].append("Rayures: verifier la presence de lignes fines dans le crop.")

    if has_any(text, ["pare choc", "pare-choc", "pare-chocs", "bumper", "calandre", "avant", "front"]):
        rules["front_bumper_logic"] = True
        rules["unknown_label"] = "front_bumper_damage"
        rules["activated_rules"].append("Avant vehicule: reclasser deformation locale en front_bumper_damage.")

    if has_any(text, ["plaque", "support plaque", "porte plaque", "plate holder"]):
        rules["plate_holder_logic"] = True
        rules["unknown_label"] = "plate_holder_damage"
        rules["activated_rules"].append("Plaque/support: autoriser la classe plate_holder_damage.")

    if has_any(text, ["inconnu", "unknown", "hors classe", "classe inconnue", "nouveau defaut"]):
        rules["allow_unknown_damage"] = True
        rules["activated_rules"].append("Hors taxonomie: utiliser unknown_damage au lieu de forcer une classe YOLO.")

    if has_any(text, ["porte", "portiere", "lateral", "cote"]) and has_any(text, ["reflet", "decor", "batiment"]):
        rules["door_reflection_skeptic"] = True
        rules["glare_reject_threshold"] = min(rules["glare_reject_threshold"], 0.38)
        rules["activated_rules"].append("Portes laterales: etre sceptique avec les reflets du decor.")

    if not rules["activated_rules"]:
        rules["activated_rules"].append("Regles par defaut: score YOLO + reflets + lignes fines + geometrie.")

    return rules


def brain_rule_rows(rules: dict) -> list[dict]:
    return [
        {"Regle": "Mode", "Valeur": rules["mode"]},
        {"Regle": "Rejeter reflets", "Valeur": str(rules["reject_reflections"])},
        {"Regle": "Rayures strictes", "Valeur": str(rules["strict_scratch"])},
        {"Regle": "Defaut inconnu autorise", "Valeur": str(rules["allow_unknown_damage"])},
        {"Regle": "Logique pare-chocs avant", "Valeur": str(rules["front_bumper_logic"])},
        {"Regle": "Logique support plaque", "Valeur": str(rules["plate_holder_logic"])},
        {"Regle": "Seuil reflet", "Valeur": f"{rules['glare_reject_threshold']:.2f}"},
        {"Regle": "Seuil rayure", "Valeur": f"{rules['scratch_confirm_threshold']:.1f}"},
        {"Regle": "Seuil bords", "Valeur": f"{rules['edge_confirm_threshold']:.3f}"},
        {"Regle": "Classe inconnue cible", "Valeur": rules["unknown_label"]},
    ]


def brain_algorithm_rows(rules: dict) -> list[dict]:
    return [
        {
            "Si": f"reflet > {rules['glare_reject_threshold']:.2f} ET lignes fines faibles",
            "Alors": "classer reflection / rejeter",
        },
        {
            "Si": f"YOLO=scratch ET carte rayure >= {rules['scratch_confirm_threshold']:.1f} ET edges >= {rules['edge_confirm_threshold']:.3f}",
            "Alors": "confirmer scratch",
        },
        {
            "Si": f"zone avant ET geometrie deformee >= {rules['geometry_reclass_threshold']:.2f}",
            "Alors": f"reclasser {rules['unknown_label']}",
        },
        {
            "Si": f"score YOLO >= {rules['confidence_confirm_threshold']:.2f} ET anomalie locale visible",
            "Alors": "confirmer la classe YOLO",
        },
        {
            "Si": "indices incomplets mais defaut possible",
            "Alors": f"mettre {rules['unknown_label']} / a_verifier",
        },
    ]


def verify_candidate_with_reasoning(det: dict, image_bgr: np.ndarray, cfg: dict, brain_rules: dict | None = None) -> dict:
    brain_rules = brain_rules or parse_reasoning_prompt(DEFAULT_BRAIN_PROMPT)
    crop_bgr, crop_box = crop_with_context(image_bgr, det["box"], padding=0.35)
    scratch_heat, _, edges = build_scratch_map(
        image_bgr,
        cfg.get("scratch_kernel", 17),
        cfg.get("scratch_threshold", 42),
        cfg.get("canny_low", 35),
        cfg.get("canny_high", 120),
    )
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    glare_mask = ((hsv[:, :, 2] >= cfg.get("glare_v", 230)) & (hsv[:, :, 1] <= cfg.get("glare_s", 80))).astype(np.uint8) * 255
    evidence = detection_visual_evidence(det, image_bgr.shape, scratch_heat, glare_mask, edges)
    crop_metrics = image_metrics(crop_bgr) if crop_bgr.size else {}
    zone = infer_vehicle_zone(det["box"], image_bgr.shape)
    geometry_score = geometry_anomaly_score(crop_bgr)
    label = det.get("label", "unknown")
    yolo_conf = float(det.get("conf", 0.0))

    reasons = []
    final_label = label
    decision = "a_verifier"
    risk = "moyen"

    glare_limit = brain_rules["glare_reject_threshold"]
    scratch_limit = brain_rules["scratch_confirm_threshold"]
    edge_limit = brain_rules["edge_confirm_threshold"]
    geometry_unknown_limit = brain_rules["geometry_unknown_threshold"]
    geometry_reclass_limit = brain_rules["geometry_reclass_threshold"]
    conf_confirm_limit = brain_rules["confidence_confirm_threshold"]

    if brain_rules["reject_reflections"] and evidence["glare_cover"] > glare_limit and evidence["scratch_p90"] < scratch_limit + 10 and yolo_conf < 0.78:
        decision = "rejeter"
        final_label = "reflection"
        risk = "haut"
        reasons.append("prompt: reflet brillant sans evidence locale suffisante")
    elif is_scratch_label(label):
        if evidence["scratch_p90"] >= scratch_limit and evidence["edge_density"] >= edge_limit and evidence["glare_cover"] < max(0.40, glare_limit):
            decision = "confirmer"
            final_label = "scratch"
            risk = "faible"
            reasons.append("prompt: lignes fines coherentes dans le crop")
        elif brain_rules["front_bumper_logic"] and geometry_score >= geometry_reclass_limit and zone.startswith("avant"):
            decision = "reclasser"
            final_label = brain_rules["unknown_label"] if brain_rules["allow_unknown_damage"] else label
            risk = "moyen"
            reasons.append("prompt: forme avant irreguliere plutot qu'une rayure")
        else:
            decision = "rejeter"
            final_label = "no_damage_or_reflection"
            risk = "haut"
            reasons.append("prompt: YOLO dit rayure mais evidence ligne faible")
    else:
        if yolo_conf >= conf_confirm_limit and geometry_score >= 0.35 and evidence["glare_cover"] < max(0.45, glare_limit):
            decision = "confirmer"
            risk = "faible"
            reasons.append("prompt: score YOLO correct et anomalie locale visible")
        elif brain_rules["front_bumper_logic"] and geometry_score >= geometry_unknown_limit and zone.startswith("avant"):
            decision = "reclasser"
            final_label = brain_rules["unknown_label"] if brain_rules["allow_unknown_damage"] else "front_bumper_damage"
            risk = "moyen"
            reasons.append("prompt: anomalie geometrique dans la zone avant")
        elif brain_rules["reject_reflections"] and evidence["glare_cover"] > min(0.38, glare_limit):
            decision = "rejeter"
            final_label = "reflection"
            risk = "haut"
            reasons.append("prompt: prediction probablement portee par un reflet")
        else:
            decision = "a_verifier"
            final_label = brain_rules["unknown_label"] if brain_rules["allow_unknown_damage"] else label
            risk = "moyen"
            reasons.append("prompt: indices insuffisants pour une classe connue")

    if brain_rules["door_reflection_skeptic"] and "portes" in zone and evidence["glare_cover"] > 0.24 and decision == "confirmer" and yolo_conf < 0.82:
        decision = "a_verifier"
        final_label = "reflection_or_damage"
        risk = "moyen"
        reasons.append("prompt: sur portiere, reflet du decor possible")

    if not reasons:
        reasons.append("decision basee sur score, reflet, lignes fines et geometrie du crop")

    return {
        **det,
        "crop": crop_bgr,
        "crop_box": crop_box,
        "vehicle_zone": zone,
        "final_label": final_label,
        "reasoning_decision": decision,
        "reasoning_risk": risk,
        "reasoning": "; ".join(reasons),
        "brain_mode": brain_rules["mode"],
        "geometry_score": geometry_score,
        "crop_contrast": crop_metrics.get("Contraste", "-"),
        "crop_sharpness": crop_metrics.get("Nettete", "-"),
        **evidence,
    }


def reasoning_rows(items: list[dict]) -> list[dict]:
    rows = []
    for idx, item in enumerate(items):
        rows.append(
            {
                "Etape": idx + 1,
                "YOLO": item.get("label", "-"),
                "Score": f"{item.get('conf', 0):.3f}",
                "Zone": item.get("vehicle_zone", "-"),
                "Decision": item.get("reasoning_decision", "-"),
                "Classe finale": item.get("final_label", "-"),
                "Risque": item.get("reasoning_risk", "-"),
                "Evidence": f"{item.get('evidence', 0):.2f}",
                "Reflet": f"{item.get('glare_cover', 0) * 100:.1f}%",
                "Raison": item.get("reasoning", "-"),
            }
        )
    return rows


def editable_reasoning_rows(items: list[dict]) -> list[dict]:
    rows = []
    for idx, item in enumerate(items):
        rows.append(
            {
                "id": idx,
                "YOLO": item.get("label", "-"),
                "Score": f"{item.get('conf', 0):.3f}",
                "Zone": item.get("vehicle_zone", "-"),
                "Decision": item.get("reasoning_decision", "a_verifier"),
                "Classe finale": item.get("final_label", "unknown_damage"),
                "Risque": item.get("reasoning_risk", "moyen"),
                "Raison": item.get("reasoning", ""),
            }
        )
    return rows


def apply_manual_reasoning_edits(items: list[dict], edited_rows: list[dict]) -> list[dict]:
    updated = [dict(item) for item in items]
    for row in edited_rows:
        try:
            idx = int(row.get("id", -1))
        except Exception:
            continue
        if idx < 0 or idx >= len(updated):
            continue
        updated[idx]["reasoning_decision"] = row.get("Decision", updated[idx].get("reasoning_decision"))
        updated[idx]["final_label"] = row.get("Classe finale", updated[idx].get("final_label"))
        updated[idx]["reasoning_risk"] = row.get("Risque", updated[idx].get("reasoning_risk"))
        updated[idx]["reasoning"] = row.get("Raison", updated[idx].get("reasoning"))
        updated[idx]["manual_reviewed"] = True
    return updated


def build_pipeline_summary(reasoned_items: list[dict]) -> dict:
    confirmed = [item for item in reasoned_items if item.get("reasoning_decision") == "confirmer"]
    reclassified = [item for item in reasoned_items if item.get("reasoning_decision") == "reclasser"]
    rejected = [item for item in reasoned_items if item.get("reasoning_decision") == "rejeter"]
    unknown = [item for item in reasoned_items if item.get("final_label") in {"unknown_damage", "front_bumper_damage"}]
    return {
        "confirmed": confirmed,
        "reclassified": reclassified,
        "rejected": rejected,
        "unknown": unknown,
        "final_items": confirmed + reclassified,
    }


def draw_reasoning_results(image_bgr: np.ndarray, items: list[dict]) -> np.ndarray:
    canvas = image_bgr.copy()
    color_map = {
        "confirmer": (64, 220, 160),
        "reclasser": (80, 190, 255),
        "a_verifier": (40, 210, 255),
        "rejeter": (120, 120, 120),
    }
    for idx, item in enumerate(items):
        x1, y1, x2, y2 = item["box"]
        decision = item.get("reasoning_decision", "a_verifier")
        color = color_map.get(decision, (80, 160, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{idx + 1} {item.get('final_label', item.get('label', '-'))}"
        cv2.putText(canvas, caption, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def draw_predictions(image_bgr: np.ndarray, result) -> tuple[np.ndarray, list[dict]]:
    canvas = image_bgr.copy()
    rows = []
    names = getattr(result, "names", {}) or {}
    for idx, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = str(names.get(cls_id, f"class_{cls_id}"))
        color = (64, 220, 160) if "scratch" in label.lower() else (80, 160, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {conf:.2f}"
        cv2.putText(canvas, caption, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        rows.append({"#": idx + 1, "Classe": label, "Score": f"{conf:.3f}", "Boite": f"{x1},{y1},{x2},{y2}"})
    return canvas, rows


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.3rem; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
    .vl-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .vl-subtitle { color: #94a3b8; margin-bottom: 1.1rem; }
    .vl-band {
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(15, 23, 42, 0.62);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .vl-pill {
        display: inline-block;
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #7dd3fc;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='vl-title'>Vision Lab - rayures fines et reflets</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='vl-subtitle'>Plateforme independante pour tester les filtres image avant detection.</div>",
    unsafe_allow_html=True,
)

with st.expander("Parametres automatiques", expanded=False):
    st.header("Pipeline vision")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    base_cfg = dict(PRESETS[preset_name])

    max_side = st.slider("Taille max de travail", 640, 2200, 1400, 100)
    auto_pilot = st.checkbox(
        "Auto-pilote parametres",
        value=True,
        help="Analyse l'image chargee et choisit automatiquement contraste, anti-reflet, nettete, carte rayures et resolution.",
    )
    if auto_pilot:
        st.caption("Auto actif: les reglages ci-dessous servent de valeurs de secours avant chargement.")

    st.subheader("Corrections")
    base_cfg["white_balance"] = st.checkbox("Balance des blancs", value=base_cfg["white_balance"], key=f"wb_{preset_name}")
    base_cfg["glare"] = st.checkbox("Attenuation des reflets", value=base_cfg["glare"], key=f"glare_{preset_name}")
    if base_cfg["glare"]:
        base_cfg["glare_v"] = st.slider("Seuil luminosite reflet", 180, 255, int(base_cfg["glare_v"]), 1, key=f"gv_{preset_name}")
        base_cfg["glare_s"] = st.slider("Seuil saturation reflet", 0, 160, int(base_cfg["glare_s"]), 1, key=f"gs_{preset_name}")
        base_cfg["glare_strength"] = st.slider("Force anti-reflet", 0.0, 0.9, float(base_cfg["glare_strength"]), 0.05, key=f"gstr_{preset_name}")

    base_cfg["denoise"] = st.slider("Denoising bilateral", 0, 12, int(base_cfg["denoise"]), 1, key=f"den_{preset_name}")
    base_cfg["clahe"] = st.checkbox("CLAHE contraste local", value=base_cfg["clahe"], key=f"clahe_{preset_name}")
    if base_cfg["clahe"]:
        base_cfg["clahe_clip"] = st.slider("CLAHE clip", 1.0, 6.0, float(base_cfg["clahe_clip"]), 0.1, key=f"clip_{preset_name}")
        base_cfg["clahe_grid"] = st.slider("CLAHE grille", 4, 16, int(base_cfg["clahe_grid"]), 1, key=f"grid_{preset_name}")

    base_cfg["gamma"] = st.slider("Gamma", 0.45, 1.80, float(base_cfg["gamma"]), 0.05, key=f"gamma_{preset_name}")
    base_cfg["detail"] = st.checkbox("Detail enhance", value=base_cfg["detail"], key=f"detail_{preset_name}")
    base_cfg["unsharp"] = st.checkbox("Nettete unsharp", value=base_cfg["unsharp"], key=f"unsharp_{preset_name}")
    if base_cfg["unsharp"]:
        base_cfg["unsharp_amount"] = st.slider("Force nettete", 0.0, 2.5, float(base_cfg["unsharp_amount"]), 0.05, key=f"ua_{preset_name}")
        base_cfg["unsharp_sigma"] = st.slider("Rayon nettete", 0.3, 3.0, float(base_cfg["unsharp_sigma"]), 0.1, key=f"us_{preset_name}")

    st.subheader("Carte rayures")
    base_cfg["scratch_kernel"] = st.slider("Longueur lignes", 5, 45, int(base_cfg["scratch_kernel"]), 2, key=f"sk_{preset_name}")
    base_cfg["scratch_threshold"] = st.slider("Seuil carte rayures", 10, 120, int(base_cfg["scratch_threshold"]), 1, key=f"sth_{preset_name}")
    base_cfg["canny_low"] = st.slider("Canny bas", 5, 120, int(base_cfg["canny_low"]), 5, key=f"cl_{preset_name}")
    base_cfg["canny_high"] = st.slider("Canny haut", 50, 260, int(base_cfg["canny_high"]), 5, key=f"ch_{preset_name}")

    st.subheader("3 filtres comparatifs")
    pseudo_blur = st.slider("Photometric stereo - lissage lumiere", 15, 95, 41, 2)
    pseudo_normal_strength = st.slider("Photometric stereo - relief", 0.5, 5.0, 2.2, 0.1)
    pseudo_light_angle = st.slider("Photometric stereo - angle lumiere", 0, 359, 315, 1)
    sr_scale = st.select_slider("Resolution scale", options=[2, 3, 4], value=2)
    use_real_esrgan = st.checkbox("Utiliser Real-ESRGAN si disponible", value=False)
    default_realesrgan_weight = find_existing_realesrgan_weight()
    realesrgan_weights_path = st.text_input("Poids Real-ESRGAN .pth", value=default_realesrgan_weight)
    realesrgan_tile = st.slider("Real-ESRGAN tile", 0, 512, 128, 32)


upload_col, status_col = st.columns([2.3, 1])
with upload_col:
    uploaded = st.file_uploader("Image vehicule", type=["jpg", "jpeg", "png", "webp", "bmp"])
with status_col:
    st.markdown("<div class='vl-band'>", unsafe_allow_html=True)
    st.markdown("<span class='vl-pill'>Independant</span><span class='vl-pill'>3 filtres</span><span class='vl-pill'>Real-ESRGAN optionnel</span>", unsafe_allow_html=True)
    st.write(f"Preset actif: **{preset_name}**")
    st.write(f"Taille max: **{max_side}px**")
    st.markdown("</div>", unsafe_allow_html=True)


if uploaded is None:
    st.info("Charge une photo de carrosserie pour comparer: photometric stereo, rayures fines, et resolution.")
    st.subheader("Priorites CNN pour ce projet")
    st.dataframe(
        [{"Concept": row[0], "Utilite": row[1], "Priorite": row[2]} for row in CONCEPT_ROWS],
        use_container_width=True,
        hide_index=True,
    )
    st.stop()


source_bgr = decode_upload(uploaded)
if source_bgr is None:
    st.error("Image illisible. Essaie un JPG, PNG, WEBP ou BMP valide.")
    st.stop()

source_bgr = resize_max_side(source_bgr, max_side)
auto_profile = None
auto_reasons = []
if auto_pilot:
    base_cfg, auto_filter_params, auto_profile, auto_reasons = auto_tune_parameters(
        source_bgr,
        base_cfg,
        sr_scale,
        realesrgan_weights_path,
    )
    pseudo_blur = auto_filter_params["pseudo_blur"]
    pseudo_normal_strength = auto_filter_params["pseudo_normal_strength"]
    pseudo_light_angle = auto_filter_params["pseudo_light_angle"]
    sr_scale = auto_filter_params["sr_scale"]
    use_real_esrgan = auto_filter_params["use_real_esrgan"]

enhanced_bgr, diagnostics = process_image(source_bgr, base_cfg)
yolo_safe_bgr, yolo_safe_diagnostics = apply_yolo_safe_enhancement(source_bgr, base_cfg, auto_profile)

original_metrics = image_metrics(source_bgr)
enhanced_metrics = image_metrics(enhanced_bgr)

view = st.radio(
    "Vue",
    ["Pipeline raisonnement", "Detection YOLO"],
    horizontal=True,
)

if auto_pilot and auto_profile:
    st.markdown("<div class='vl-band'>", unsafe_allow_html=True)
    st.write("**Auto-pilote actif** - decisions: " + ", ".join(auto_reasons))
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("Voir le diagnostic et les parametres auto", expanded=False):
        profile_rows = [
            {"Mesure": "Luminosite", "Valeur": f"{auto_profile['brightness']:.1f}"},
            {"Mesure": "Contraste", "Valeur": f"{auto_profile['contrast']:.1f}"},
            {"Mesure": "Nettete", "Valeur": f"{auto_profile['sharpness']:.1f}"},
            {"Mesure": "Bruit estime", "Valeur": f"{auto_profile['noise']:.1f}"},
            {"Mesure": "Reflets", "Valeur": f"{auto_profile['glare_ratio']:.2f}%"},
            {"Mesure": "Hautes lumieres", "Valeur": f"{auto_profile['highlight_ratio']:.2f}%"},
            {"Mesure": "Reponse lignes fines", "Valeur": f"{auto_profile['scratch_response']:.1f}"},
        ]
        st.dataframe(profile_rows, use_container_width=True, hide_index=True)
        auto_rows = [
            {"Parametre": "Anti-reflet", "Valeur": str(base_cfg["glare"])},
            {"Parametre": "CLAHE clip", "Valeur": f"{base_cfg['clahe_clip']:.1f}"},
            {"Parametre": "Gamma", "Valeur": f"{base_cfg['gamma']:.2f}"},
            {"Parametre": "Denoising", "Valeur": str(base_cfg["denoise"])},
            {"Parametre": "Nettete", "Valeur": f"{base_cfg['unsharp_amount']:.2f}"},
            {"Parametre": "Kernel rayures", "Valeur": str(base_cfg["scratch_kernel"])},
            {"Parametre": "Seuil rayures", "Valeur": str(base_cfg["scratch_threshold"])},
            {"Parametre": "Photometric relief", "Valeur": f"{pseudo_normal_strength:.1f}"},
            {"Parametre": "Resolution scale", "Valeur": f"x{sr_scale}"},
        ]
        st.dataframe(auto_rows, use_container_width=True, hide_index=True)

if view == "Pipeline raisonnement":
    st.markdown("<div class='vl-band'>", unsafe_allow_html=True)
    st.write(
        "Pipeline en 5 etapes: profil image -> image YOLO-safe -> candidats YOLO/zoom -> crops -> verification raisonnee. "
        "Chaque etape nourrit la suivante."
    )
    st.info(
        "VLM reel non integre pour l'instant: cette version utilise un cerveau local prompt -> regles. "
        "Le prochain niveau sera de brancher un VLM professeur pour generer ces decisions avec une analyse visuelle plus riche."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns([2, 1, 1])
    with p1:
        default_model = "best_2.pt" if os.path.exists("best_2.pt") else "best.pt"
        pipeline_model_path = st.text_input("Modele YOLO du pipeline", value=default_model, key="pipeline_model_path")
    with p2:
        pipeline_conf = st.slider("Confiance pipeline", 0.05, 0.95, 0.32, 0.05, key="pipeline_conf")
    with p3:
        pipeline_iou = st.slider("IoU pipeline", 0.10, 0.90, 0.45, 0.05, key="pipeline_iou")

    z1, z2, z3, z4 = st.columns(4)
    with z1:
        pipeline_tile_size = st.select_slider("Zone zoom", options=[384, 512, 640, 768, 896], value=640, key="pipeline_tile_size")
    with z2:
        pipeline_overlap = st.slider("Overlap zoom", 0.10, 0.55, 0.30, 0.05, key="pipeline_overlap")
    with z3:
        pipeline_zoom = st.select_slider("Facteur zoom", options=[1.0, 1.5, 2.0, 2.5], value=2.0, key="pipeline_zoom")
    with z4:
        pipeline_imgsz = st.select_slider("imgsz", options=[640, 832, 960, 1280], value=960, key="pipeline_imgsz")
    pipeline_max_tiles = st.slider("Zones max", 4, 40, 16, 2, key="pipeline_max_tiles")

    st.subheader("Cerveau prompt - ton raisonnement")
    brain_prompt = st.text_area(
        "Ecris comment l'IA doit raisonner",
        value=DEFAULT_BRAIN_PROMPT,
        height=170,
        key="brain_prompt",
        help="Ce texte est converti localement en regles qui guident la verification des crops.",
    )
    brain_rules = parse_reasoning_prompt(brain_prompt)
    r1, r2 = st.columns([1, 1.2])
    with r1:
        st.caption("Regles extraites du prompt")
        st.dataframe(brain_rule_rows(brain_rules), use_container_width=True, hide_index=True)
    with r2:
        st.caption("Algorithme genere")
        st.dataframe(brain_algorithm_rows(brain_rules), use_container_width=True, hide_index=True)
    with st.expander("Intentions detectees dans ton prompt"):
        for activated_rule in brain_rules["activated_rules"]:
            st.write(f"- {activated_rule}")

    st.subheader("Etape 1 - Profil de l'image")
    profile = auto_profile or image_profile(source_bgr)
    st.dataframe(
        [
            {"Signal": "Luminosite", "Valeur": f"{profile['brightness']:.1f}"},
            {"Signal": "Contraste", "Valeur": f"{profile['contrast']:.1f}"},
            {"Signal": "Nettete", "Valeur": f"{profile['sharpness']:.1f}"},
            {"Signal": "Bruit", "Valeur": f"{profile['noise']:.1f}"},
            {"Signal": "Reflets", "Valeur": f"{profile['glare_ratio']:.2f}%"},
            {"Signal": "Reponse lignes fines", "Valeur": f"{profile['scratch_response']:.1f}"},
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Etape 2 - Image naturelle pour YOLO")
    e2a, e2b = st.columns(2)
    with e2a:
        st.caption("Original")
        st.image(to_rgb(source_bgr), use_container_width=True)
    with e2b:
        st.caption("YOLO-safe")
        st.image(to_rgb(yolo_safe_bgr), use_container_width=True)

    if st.button("Lancer le pipeline raisonne", type="primary", use_container_width=True):
        if YOLO is None:
            st.error("Ultralytics n'est pas installe dans cet environnement.")
        elif not os.path.exists(pipeline_model_path):
            st.error(f"Modele introuvable: {pipeline_model_path}")
        else:
            detector = load_yolo_model(pipeline_model_path, os.path.getmtime(pipeline_model_path))
            zoom_conf = max(0.12, pipeline_conf - 0.08)

            with st.spinner("Etape 3 - Generation des candidats YOLO global + zoom..."):
                original_result = detector(to_pil(source_bgr), conf=pipeline_conf, iou=pipeline_iou, imgsz=int(pipeline_imgsz), verbose=False)[0]
                safe_result = detector(to_pil(yolo_safe_bgr), conf=pipeline_conf, iou=pipeline_iou, imgsz=int(pipeline_imgsz), verbose=False)[0]
                original_dets = extract_detections(original_result, "original global")
                safe_dets = extract_detections(safe_result, "yolo-safe global")
                zoom_dets, selected_tiles = run_tiled_zoom_detection(
                    detector,
                    yolo_safe_bgr,
                    base_cfg,
                    zoom_conf,
                    pipeline_iou,
                    pipeline_tile_size,
                    pipeline_overlap,
                    pipeline_zoom,
                    pipeline_imgsz,
                    pipeline_max_tiles,
                    "yolo-safe",
                )
                candidate_dets = merge_duplicate_detections(original_dets + safe_dets + zoom_dets, iou_threshold=0.34)

            st.subheader("Etape 3 - Candidats proposes")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Global original", len(original_dets))
            c2.metric("Global YOLO-safe", len(safe_dets))
            c3.metric("Zoom local", len(zoom_dets))
            c4.metric("Candidats fusionnes", len(candidate_dets))
            st.image(to_rgb(draw_tiles(source_bgr, selected_tiles)), caption="Zones analysees par zoom", use_container_width=True)
            if candidate_dets:
                st.dataframe(detection_rows(candidate_dets), use_container_width=True, hide_index=True)
            else:
                st.warning("Aucun candidat YOLO. Le pipeline ne peut pas raisonner sans zone candidate; baisse la confiance ou augmente les zones.")

            with st.spinner("Etape 4 - Verification des crops et raisonnement..."):
                reasoned_items = [verify_candidate_with_reasoning(det, source_bgr, base_cfg, brain_rules) for det in candidate_dets]
                summary = build_pipeline_summary(reasoned_items)
                st.session_state.brain_last_reasoned_items = reasoned_items
                st.session_state.brain_last_candidates = candidate_dets
                st.session_state.brain_last_prompt = brain_prompt
                st.session_state.brain_last_rules = brain_rules
                st.session_state.brain_last_source_filename = uploaded.name if uploaded is not None else "uploaded_image"

            st.subheader("Etape 4 - Crops verifies")
            if reasoned_items:
                crop_cols = st.columns(min(4, len(reasoned_items)))
                for idx, item in enumerate(reasoned_items[:8]):
                    with crop_cols[idx % len(crop_cols)]:
                        st.caption(f"#{idx + 1} {item['reasoning_decision']} -> {item['final_label']}")
                        st.image(to_rgb(item["crop"]), use_container_width=True)
                st.dataframe(reasoning_rows(reasoned_items), use_container_width=True, hide_index=True)
            else:
                st.info("Aucun crop a verifier.")

            st.subheader("Etape 5 - Decision finale")
            final_canvas = draw_reasoning_results(source_bgr, summary["final_items"])
            st.image(to_rgb(final_canvas), use_container_width=True)
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Confirmes", len(summary["confirmed"]))
            f2.metric("Reclasses", len(summary["reclassified"]))
            f3.metric("Rejetes", len(summary["rejected"]))
            f4.metric("Inconnus", len(summary["unknown"]))

            if summary["final_items"]:
                st.success("Le pipeline a produit une decision finale exploitable.")
                st.dataframe(reasoning_rows(summary["final_items"]), use_container_width=True, hide_index=True)
            else:
                st.warning("Aucun defaut confirme. Les predictions ressemblent plutot a des reflets/no-damage ou sont trop faibles.")

            if summary["unknown"]:
                st.info(
                    "Des zones semblent hors taxonomie YOLO. Ajoute des exemples dans une classe `unknown_damage`, "
                    "`front_bumper_damage` ou `bumper_damage` avant de reentrainer."
                )

    last_items = st.session_state.get("brain_last_reasoned_items", [])
    if last_items:
        st.subheader("Correction manuelle du resultat")
        st.write("Tu peux modifier la decision finale avant de sauvegarder les exemples du cerveau permanent.")
        edited_rows = st.data_editor(
            editable_reasoning_rows(last_items),
            use_container_width=True,
            hide_index=True,
            disabled=["id", "YOLO", "Score", "Zone"],
            column_config={
                "Decision": st.column_config.SelectboxColumn(
                    "Decision",
                    options=["confirmer", "rejeter", "reclasser", "a_verifier"],
                ),
                "Risque": st.column_config.SelectboxColumn(
                    "Risque",
                    options=["faible", "moyen", "haut"],
                ),
            },
            key="brain_manual_editor",
        )
        if st.button("Appliquer mes corrections", use_container_width=True):
            corrected_items = apply_manual_reasoning_edits(last_items, edited_rows)
            st.session_state.brain_last_reasoned_items = corrected_items
            st.success("Corrections appliquees. Le dataset cerveau utilisera ces valeurs.")
            st.rerun()

        corrected_summary = build_pipeline_summary(st.session_state.get("brain_last_reasoned_items", []))
        st.subheader("YOLO seul vs YOLO + raisonnement")
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.caption("Candidats YOLO avant raisonnement")
            st.image(to_rgb(draw_detection_list(source_bgr, st.session_state.get("brain_last_candidates", []))), use_container_width=True)
        with cmp2:
            st.caption("Resultat apres raisonnement/corrections")
            st.image(to_rgb(draw_reasoning_results(source_bgr, corrected_summary["final_items"])), use_container_width=True)

    st.subheader("Dataset du cerveau permanent")
    dataset_summary = summarize_brain_dataset()
    if dataset_summary["exists"]:
        st.write(f"Exemples enregistres: **{dataset_summary['total']}** dans `{dataset_summary['root']}`")
        if dataset_summary["labels"]:
            st.dataframe(dataset_summary["labels"], use_container_width=True, hide_index=True)
    else:
        st.info("Aucun exemple enregistre pour le cerveau permanent pour l'instant.")

    if last_items:
        include_rejected = st.checkbox(
            "Inclure aussi les rejets/reflets/no_damage",
            value=True,
            help="Tres utile pour apprendre au cerveau a supprimer les faux positifs.",
        )
        if st.button("Enregistrer les crops du dernier pipeline", use_container_width=True):
            save_info = save_brain_training_samples(
                last_items,
                st.session_state.get("brain_last_source_filename", "uploaded_image"),
                st.session_state.get("brain_last_prompt", brain_prompt),
                st.session_state.get("brain_last_rules", brain_rules),
                include_rejected,
            )
            st.success(
                f"{save_info['saved_count']} exemples ajoutes au dataset cerveau. "
                f"Classes: {', '.join(save_info['labels']) if save_info['labels'] else 'aucune'}"
            )
            st.write(f"Manifest: `{save_info['manifest_path']}`")
    else:
        st.caption("Lance d'abord le pipeline raisonne pour generer des crops enregistrables.")

elif view == "Comparaison 3 filtres":
    st.markdown("<div class='vl-band'>", unsafe_allow_html=True)
    st.write(
        "Le photometric stereo exact demande plusieurs photos avec des lumieres differentes. "
        "Ici, le filtre utilise une approximation monoculaire pour separer illumination, relief et reflet."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    pseudo_bgr, pseudo_diag = apply_pseudo_photometric_stereo(
        source_bgr,
        blur_size=pseudo_blur,
        normal_strength=pseudo_normal_strength,
        light_angle=pseudo_light_angle,
        glare_value=base_cfg.get("glare_v", 225),
        glare_saturation=base_cfg.get("glare_s", 80),
    )
    scratch_bgr, scratch_diag = apply_scratch_visibility_filter(source_bgr, base_cfg)
    sr_bgr, sr_status = apply_realesrgan_or_fallback(
        source_bgr,
        scale=sr_scale,
        weights_path=realesrgan_weights_path,
        tile=realesrgan_tile,
        use_real_esrgan=use_real_esrgan,
    )

    if "Fallback" in sr_status:
        st.warning(sr_status)
    else:
        st.success(sr_status)

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.subheader("Original")
        st.image(to_rgb(source_bgr), use_container_width=True)
        st.caption(original_metrics["Resolution"])
    with c1:
        st.subheader("Photometric stereo")
        st.image(to_rgb(pseudo_bgr), use_container_width=True)
        st.caption("Reflets/relief de surface")
    with c2:
        st.subheader("Rayures fines")
        st.image(to_rgb(scratch_bgr), use_container_width=True)
        st.caption("Micro-contraste + lignes candidates")
    with c3:
        st.subheader("Resolution")
        st.image(to_rgb(sr_bgr), use_container_width=True)
        st.caption(f"{image_metrics(sr_bgr)['Resolution']} - {sr_status}")

    metrics_rows = []
    for label, image_bgr in [
        ("Original", source_bgr),
        ("Photometric stereo", pseudo_bgr),
        ("Rayures fines", scratch_bgr),
        ("Resolution", sr_bgr),
    ]:
        row = {"Filtre": label}
        row.update(image_metrics(image_bgr))
        metrics_rows.append(row)
    st.dataframe(metrics_rows, use_container_width=True, hide_index=True)

    with st.expander("Diagnostics des 3 filtres"):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.caption("Masque reflet")
            st.image(pseudo_diag["glare_mask"], clamp=True, use_container_width=True)
        with d2:
            st.caption("Reflectance")
            st.image(pseudo_diag["reflectance"], clamp=True, use_container_width=True)
        with d3:
            st.caption("Carte rayures")
            st.image(cv2.applyColorMap(scratch_diag["scratch_heat"], cv2.COLORMAP_TURBO), channels="BGR", use_container_width=True)
        with d4:
            st.caption("Masque rayures")
            st.image(scratch_diag["scratch_binary"], clamp=True, use_container_width=True)

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "Telecharger photometric PNG",
            data=encode_png(pseudo_bgr),
            file_name=f"{Path(uploaded.name).stem}_photometric.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Telecharger rayures PNG",
            data=encode_png(scratch_bgr),
            file_name=f"{Path(uploaded.name).stem}_scratch_filter.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl3:
        st.download_button(
            "Telecharger resolution PNG",
            data=encode_png(sr_bgr),
            file_name=f"{Path(uploaded.name).stem}_sr_x{sr_scale}.png",
            mime="image/png",
            use_container_width=True,
        )

elif view == "Comparaison":
    st.markdown("<div class='vl-band'>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Contraste", enhanced_metrics["Contraste"], metric_delta(original_metrics, enhanced_metrics)[0]["Delta"])
    m2.metric("Nettete", enhanced_metrics["Nettete"], metric_delta(original_metrics, enhanced_metrics)[1]["Delta"])
    m3.metric("Pixels reflet", enhanced_metrics["Pixels reflet"], metric_delta(original_metrics, enhanced_metrics)[2]["Delta"])
    m4.metric("Entropie", enhanced_metrics["Entropie"], metric_delta(original_metrics, enhanced_metrics)[3]["Delta"])
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(to_rgb(source_bgr), use_container_width=True)
        st.table(original_metrics)
    with c2:
        st.subheader("Apres filtres")
        st.image(to_rgb(enhanced_bgr), use_container_width=True)
        st.table(enhanced_metrics)

    st.download_button(
        "Telecharger image filtree PNG",
        data=encode_png(enhanced_bgr),
        file_name=f"{Path(uploaded.name).stem}_vision_lab.png",
        mime="image/png",
        use_container_width=True,
    )

elif view == "Diagnostics rayures/reflets":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Carte rayures")
        heat_color = cv2.applyColorMap(diagnostics["scratch_heat"], cv2.COLORMAP_TURBO)
        st.image(to_rgb(heat_color), use_container_width=True)
    with c2:
        st.subheader("Masque candidats")
        st.image(diagnostics["scratch_binary"], clamp=True, use_container_width=True)
    with c3:
        st.subheader("Masque reflets")
        st.image(diagnostics["glare_mask"], clamp=True, use_container_width=True)

    overlay = enhanced_bgr.copy()
    overlay[diagnostics["scratch_binary"] > 0] = (0, 255, 255)
    blended = cv2.addWeighted(enhanced_bgr, 0.72, overlay, 0.28, 0)
    st.subheader("Overlay candidats sur image filtree")
    st.image(to_rgb(blended), use_container_width=True)

elif view == "Detection YOLO":
    st.markdown("<div class='vl-band'>", unsafe_allow_html=True)
    st.write(
        "Mode IA robuste: les filtres agressifs servent au diagnostic visuel, pas comme entree principale du modele. "
        "Pour les petits defauts, le mode zoom local decoupe l'image en zones chevauchees puis remappe les boites."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    y1, y2, y3 = st.columns([2, 1, 1])
    with y1:
        default_model = "best_2.pt" if os.path.exists("best_2.pt") else "best.pt"
        model_path = st.text_input("Chemin modele .pt", value=default_model)
    with y2:
        yolo_conf = st.slider("Confiance minimale", 0.05, 0.95, 0.35, 0.05)
    with y3:
        yolo_iou = st.slider("IoU", 0.10, 0.90, 0.45, 0.05)

    detection_mode = st.radio(
        "Strategie detection",
        [
            "Zoom local par zones (recommande rayures)",
            "Ensemble stable (recommande)",
            "Original seulement",
            "YOLO-safe seulement",
            "Filtres agressifs (diagnostic, non recommande)",
        ],
        horizontal=True,
    )
    post_filter = st.checkbox("Anti-faux positifs", value=True)

    if detection_mode == "Zoom local par zones (recommande rayures)":
        z1, z2, z3, z4 = st.columns(4)
        with z1:
            zoom_tile_size = st.select_slider("Taille zone", options=[384, 512, 640, 768, 896], value=640)
        with z2:
            zoom_overlap = st.slider("Chevauchement", 0.10, 0.55, 0.30, 0.05)
        with z3:
            zoom_factor = st.select_slider("Zoom zone", options=[1.0, 1.5, 2.0, 2.5], value=2.0)
        with z4:
            yolo_imgsz = st.select_slider("YOLO imgsz", options=[640, 832, 960, 1280], value=960)
        max_zoom_tiles = st.slider("Nombre max de zones analysees", 4, 40, 16, 2)

    with st.expander("Voir l'image YOLO-safe"):
        s1, s2 = st.columns(2)
        with s1:
            st.caption("Original")
            st.image(to_rgb(source_bgr), use_container_width=True)
        with s2:
            st.caption("YOLO-safe: correction douce sans artefacts")
            st.image(to_rgb(yolo_safe_bgr), use_container_width=True)

    if st.button("Lancer detection", type="primary", use_container_width=True):
        if YOLO is None:
            st.error("Ultralytics n'est pas installe dans cet environnement.")
        elif not os.path.exists(model_path):
            st.error(f"Modele introuvable: {model_path}")
        else:
            model_mtime = os.path.getmtime(model_path)
            detector = load_yolo_model(model_path, model_mtime)

            if detection_mode == "Zoom local par zones (recommande rayures)":
                zoom_conf = max(0.12, yolo_conf - 0.08)
                with st.spinner("Detection globale + zoom local sur zones candidates..."):
                    original_result = detector(to_pil(source_bgr), conf=yolo_conf, iou=yolo_iou, imgsz=int(yolo_imgsz), verbose=False)[0]
                    safe_result = detector(to_pil(yolo_safe_bgr), conf=yolo_conf, iou=yolo_iou, imgsz=int(yolo_imgsz), verbose=False)[0]
                    original_dets = extract_detections(original_result, "original global")
                    safe_dets = extract_detections(safe_result, "yolo-safe global")
                    zoom_dets, selected_tiles = run_tiled_zoom_detection(
                        detector,
                        yolo_safe_bgr,
                        base_cfg,
                        zoom_conf,
                        yolo_iou,
                        zoom_tile_size,
                        zoom_overlap,
                        zoom_factor,
                        yolo_imgsz,
                        max_zoom_tiles,
                        "yolo-safe",
                    )

                    if post_filter:
                        global_dets, global_rejected, ensemble_diag = stable_ensemble_detections(
                            original_dets,
                            safe_dets,
                            source_bgr,
                            yolo_safe_bgr,
                            base_cfg,
                            yolo_conf,
                        )
                        zoom_kept, zoom_rejected, zoom_diag = validate_detections(zoom_dets, yolo_safe_bgr, base_cfg, zoom_conf)
                        promoted_zoom = []
                        for det in zoom_kept:
                            strong_scratch = is_scratch_label(det["label"]) and det.get("evidence", 0) >= 0.42 and det.get("glare_cover", 0) < 0.38
                            strong_conf = det["conf"] >= yolo_conf + 0.05 and det.get("glare_cover", 0) < 0.45
                            if strong_scratch or strong_conf:
                                det = dict(det)
                                det["decision"] = "gardee par zoom local"
                                promoted_zoom.append(det)
                            else:
                                det = dict(det)
                                det["status"] = "rejetee"
                                det["decision"] = "zoom faible ou instable"
                                zoom_rejected.append(det)
                        final_dets = merge_duplicate_detections(global_dets + promoted_zoom, iou_threshold=0.34)
                        rejected_dets = global_rejected + zoom_rejected
                    else:
                        final_dets = merge_duplicate_detections(original_dets + safe_dets + zoom_dets, iou_threshold=0.34)
                        rejected_dets = []
                        ensemble_diag = {}
                        zoom_diag = {}

                drawn = draw_detection_list(source_bgr, final_dets)
                st.subheader("Resultat zoom local")
                st.image(to_rgb(drawn), use_container_width=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Zones analysees", len(selected_tiles))
                m2.metric("Detections gardees", len(final_dets))
                m3.metric("Predictions rejetees", len(rejected_dets))
                if final_dets:
                    st.dataframe(detection_rows(final_dets), use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune detection robuste. Tu peux augmenter `Nombre max de zones`, `YOLO imgsz` ou baisser legerement la confiance.")
                if rejected_dets:
                    with st.expander("Predictions rejetees"):
                        st.dataframe(detection_rows(rejected_dets), use_container_width=True, hide_index=True)
                with st.expander("Zones zoomees analysees"):
                    st.image(to_rgb(draw_tiles(source_bgr, selected_tiles)), use_container_width=True)
                if post_filter:
                    with st.expander("Diagnostics zoom / anti-faux positifs"):
                        d1, d2, d3 = st.columns(3)
                        with d1:
                            st.caption("Carte rayures sur YOLO-safe")
                            st.image(cv2.applyColorMap(zoom_diag["scratch_heat"], cv2.COLORMAP_TURBO), channels="BGR", use_container_width=True)
                        with d2:
                            st.caption("Masque reflets")
                            st.image(zoom_diag["glare_mask"], clamp=True, use_container_width=True)
                        with d3:
                            st.caption("Bords")
                            st.image(zoom_diag["edges"], clamp=True, use_container_width=True)

            elif detection_mode == "Ensemble stable (recommande)":
                with st.spinner("Detection originale + YOLO-safe, puis validation anti-faux positifs..."):
                    original_result = detector(to_pil(source_bgr), conf=yolo_conf, iou=yolo_iou, verbose=False)[0]
                    safe_result = detector(to_pil(yolo_safe_bgr), conf=yolo_conf, iou=yolo_iou, verbose=False)[0]
                    original_dets = extract_detections(original_result, "original")
                    safe_dets = extract_detections(safe_result, "yolo-safe")
                    if post_filter:
                        final_dets, rejected_dets, ensemble_diag = stable_ensemble_detections(
                            original_dets,
                            safe_dets,
                            source_bgr,
                            yolo_safe_bgr,
                            base_cfg,
                            yolo_conf,
                        )
                    else:
                        final_dets = merge_duplicate_detections(original_dets + safe_dets)
                        rejected_dets = []
                        ensemble_diag = {}

                drawn = draw_detection_list(source_bgr, final_dets)
                st.subheader("Resultat stable")
                st.image(to_rgb(drawn), use_container_width=True)
                st.metric("Detections gardees", len(final_dets))
                st.metric("Predictions rejetees", len(rejected_dets))
                if final_dets:
                    st.dataframe(detection_rows(final_dets), use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune detection stable. C'est souvent mieux que des faux positifs sur reflet.")
                if rejected_dets:
                    with st.expander("Predictions rejetees"):
                        st.dataframe(detection_rows(rejected_dets), use_container_width=True, hide_index=True)
                if ensemble_diag:
                    with st.expander("Diagnostics anti-faux positifs"):
                        d1, d2, d3 = st.columns(3)
                        with d1:
                            st.caption("Evidence rayures YOLO-safe")
                            st.image(cv2.applyColorMap(ensemble_diag["yolo_safe"]["scratch_heat"], cv2.COLORMAP_TURBO), channels="BGR", use_container_width=True)
                        with d2:
                            st.caption("Masque reflets YOLO-safe")
                            st.image(ensemble_diag["yolo_safe"]["glare_mask"], clamp=True, use_container_width=True)
                        with d3:
                            st.caption("Edges YOLO-safe")
                            st.image(ensemble_diag["yolo_safe"]["edges"], clamp=True, use_container_width=True)
            else:
                if detection_mode == "Original seulement":
                    target_label = "Original"
                    target_image = source_bgr
                elif detection_mode == "YOLO-safe seulement":
                    target_label = "YOLO-safe"
                    target_image = yolo_safe_bgr
                else:
                    target_label = "Filtres agressifs"
                    target_image = enhanced_bgr
                    st.warning("Ce mode sert seulement au diagnostic: il peut creer beaucoup de faux positifs.")

                with st.spinner(f"Detection {target_label.lower()}..."):
                    result = detector(to_pil(target_image), conf=yolo_conf, iou=yolo_iou, verbose=False)[0]
                    detections = extract_detections(result, target_label.lower())
                    if post_filter:
                        final_dets, rejected_dets, diag = validate_detections(detections, target_image, base_cfg, yolo_conf)
                    else:
                        final_dets, rejected_dets, diag = detections, [], {}

                drawn = draw_detection_list(target_image, final_dets)
                st.subheader(target_label)
                st.image(to_rgb(drawn), use_container_width=True)
                if final_dets:
                    st.dataframe(detection_rows(final_dets), use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune detection valide.")
                if rejected_dets:
                    with st.expander("Predictions rejetees"):
                        st.dataframe(detection_rows(rejected_dets), use_container_width=True, hide_index=True)

else:
    st.subheader("Priorites CNN pour rayures fines et reflets")
    st.dataframe(
        [{"Concept": row[0], "Utilite": row[1], "Priorite": row[2]} for row in CONCEPT_ROWS],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Plan d'experimentation conseille")
    st.markdown(
        """
        1. Construire un jeu test fixe avec rayures fines, reflets, poussieres et images saines.
        2. Tester les presets sans reentrainer le modele, puis noter precision, recall et faux positifs.
        3. Garder les filtres qui augmentent le recall des rayures sans transformer les reflets en defauts.
        4. Ajouter des images negatives avec reflets forts pour reduire l'overfitting.
        5. Reentrainer avec augmentations controlees: contraste, luminosite, blur leger, reflets.
        """
    )
