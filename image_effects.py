#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blender gerektirmeyen kamera ve düşük çözünürlük simülasyonları."""

from __future__ import annotations

from io import BytesIO
import math
import random
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Girdiyi kayıpsız biçimde uint8 RGB diziye dönüştürür."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("Görüntü HxWx3 veya HxWx4 olmalıdır.")
    return arr[:, :, :3]


def _motion_blur(image: np.ndarray, radius: int, angle_deg: float) -> np.ndarray:
    """Küçük kaydırmaların ortalamasıyla hafif yönlü hareket bulanıklığı."""
    if radius <= 0:
        return image
    angle = math.radians(angle_deg)
    accum = np.zeros_like(image, dtype=np.float32)
    count = 0
    for step in range(-radius, radius + 1):
        dx = int(round(math.cos(angle) * step))
        dy = int(round(math.sin(angle) * step))
        shifted = np.roll(image, shift=(dy, dx), axis=(0, 1))
        if dy > 0:
            shifted[:dy] = image[:dy]
        elif dy < 0:
            shifted[dy:] = image[dy:]
        if dx > 0:
            shifted[:, :dx] = image[:, :dx]
        elif dx < 0:
            shifted[:, dx:] = image[:, dx:]
        accum += shifted.astype(np.float32)
        count += 1
    return np.clip(accum / max(count, 1), 0, 255).astype(np.uint8)


def radial_distort_array(array: np.ndarray, coefficient: float, order: int = 1) -> np.ndarray:
    """Boyutu değiştirmeden hafif radyal lens bozulması uygular.

    Yardımcı maske/segmentasyon dizileri için ``order=0`` kullanılmalıdır.
    SciPy yoksa girdiyi değiştirmeden döndürür.
    """
    if abs(coefficient) < 1e-9:
        return np.asarray(array).copy()
    try:
        from scipy.ndimage import map_coordinates
    except Exception:
        return np.asarray(array).copy()

    src = np.asarray(array)
    h, w = src.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    scale = max(cx, cy, 1.0)
    xn = (xx - cx) / scale
    yn = (yy - cy) / scale
    r2 = xn * xn + yn * yn
    factor = 1.0 + coefficient * r2
    src_x = cx + xn * factor * scale
    src_y = cy + yn * factor * scale

    if src.ndim == 2:
        warped = map_coordinates(src, [src_y, src_x], order=order, mode="nearest")
    else:
        channels = [
            map_coordinates(src[:, :, c], [src_y, src_x], order=order, mode="nearest")
            for c in range(src.shape[2])
        ]
        warped = np.stack(channels, axis=-1)
    return warped.astype(src.dtype, copy=False)


def distort_bbox(
    bbox_xywh: tuple[float, float, float, float] | None,
    coefficient: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Radyal bozulmadan sonra eksen hizalı bbox'ı yaklaşık olarak günceller."""
    if bbox_xywh is None or abs(coefficient) < 1e-9:
        return bbox_xywh
    x, y, w, h = bbox_xywh
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    norm = max(cx, cy, 1.0)
    points = [
        (x, y), (x + w, y), (x, y + h), (x + w, y + h),
        (x + w / 2, y), (x + w / 2, y + h),
        (x, y + h / 2), (x + w, y + h / 2),
    ]
    transformed = []
    for px, py in points:
        xn, yn = (px - cx) / norm, (py - cy) / norm
        # radial_distort_array ters eşleme yaptığı için ileri eşlemenin hafif-k
        # yaklaşımı aşağıdaki bölme ile elde edilir.
        factor = 1.0 + coefficient * (xn * xn + yn * yn)
        transformed.append((cx + xn / factor * norm, cy + yn / factor * norm))
    xs = [max(0.0, min(float(width), p[0])) for p in transformed]
    ys = [max(0.0, min(float(height), p[1])) for p in transformed]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _apply_temperature(pil: Image.Image, kelvin_shift: float) -> Image.Image:
    """Nötr 6500 K çevresinde sınırlı sıcak/soğuk kanal kayması."""
    normalized = max(-1.0, min(1.0, (kelvin_shift - 6500.0) / 3500.0))
    red_factor = 1.0 - 0.10 * normalized
    blue_factor = 1.0 + 0.10 * normalized
    r, g, b = pil.split()
    r = r.point(lambda value: min(255, int(value * red_factor)))
    b = b.point(lambda value: min(255, int(value * blue_factor)))
    return Image.merge("RGB", (r, g, b))


def process_image(
    image: np.ndarray,
    config: dict[str, Any],
    rng: random.Random | Any = random,
    np_rng: np.random.Generator | None = None,
    auxiliary_arrays: Iterable[np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any], list[np.ndarray]]:
    """Kamera efektlerini uygular ve kullanılan parametreleri döndürür.

    Geometrik lens bozulması yardımcı dizilere de uygulanır. Düşük çözünürlük
    simülasyonu yalnızca görüntüyü örnekler; tuval boyutu değişmediği için YOLO
    ve COCO koordinatları aynı kalır.
    """
    arr = _as_uint8_rgb(image)
    np_rng = np_rng or np.random.default_rng()
    aux = [np.asarray(item).copy() for item in (auxiliary_arrays or [])]
    meta: dict[str, Any] = {
        "render_resolution": [int(arr.shape[1]), int(arr.shape[0])],
        "output_resolution": [int(arr.shape[1]), int(arr.shape[0])],
    }
    if not config.get("enabled", True):
        return arr, meta, aux

    pil = Image.fromarray(arr)

    if rng.random() < config.get("exposure_probability", 0.0):
        ev = rng.uniform(*config["exposure_ev_range"])
        pil = ImageEnhance.Brightness(pil).enhance(2.0 ** ev)
        meta["exposure_ev"] = round(ev, 3)

    if rng.random() < config.get("temperature_probability", 0.0):
        kelvin = rng.uniform(*config["temperature_range"])
        pil = _apply_temperature(pil, kelvin)
        meta["color_temperature_k"] = round(kelvin)

    if rng.random() < config.get("brightness_contrast_probability", 0.0):
        brightness = rng.uniform(*config["brightness_range"])
        contrast = rng.uniform(*config["contrast_range"])
        pil = ImageEnhance.Brightness(pil).enhance(brightness)
        pil = ImageEnhance.Contrast(pil).enhance(contrast)
        meta["brightness"] = round(brightness, 3)
        meta["contrast"] = round(contrast, 3)

    if rng.random() < config.get("gamma_probability", 0.0):
        gamma = rng.uniform(*config["gamma_range"])
        lut = [min(255, round(255 * ((i / 255.0) ** (1.0 / gamma)))) for i in range(256)]
        pil = pil.point(lut * 3)
        meta["gamma"] = round(gamma, 3)

    coefficient = 0.0
    if rng.random() < config.get("lens_distortion_probability", 0.0):
        coefficient = rng.uniform(*config["lens_distortion_range"])
        arr = radial_distort_array(np.asarray(pil), coefficient, order=1)
        aux = [radial_distort_array(item, coefficient, order=0) for item in aux]
        pil = Image.fromarray(arr)
        meta["lens_distortion_k"] = round(coefficient, 6)

    if rng.random() < config.get("motion_blur_probability", 0.0):
        radius = rng.randint(*config["motion_blur_radius_range"])
        angle = rng.uniform(0.0, 180.0)
        pil = Image.fromarray(_motion_blur(np.asarray(pil), radius, angle))
        meta["motion_blur"] = {"radius": radius, "angle_deg": round(angle, 2)}

    if rng.random() < config.get("gaussian_blur_probability", 0.0):
        blur = rng.uniform(*config["gaussian_blur_range"])
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur))
        meta["gaussian_blur_radius"] = round(blur, 3)

    arr = np.asarray(pil).copy()
    if rng.random() < config.get("noise_probability", 0.0):
        sigma = rng.uniform(*config["noise_sigma_range"])
        noise = np_rng.normal(0.0, sigma, arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        meta["noise_sigma"] = round(sigma, 3)
        pil = Image.fromarray(arr)

    if rng.random() < config.get("low_res_probability", 0.0):
        target = int(rng.choice(list(config["low_res_sizes"])))
        method = str(rng.choice(list(config["upscale_methods"]))).lower()
        resample = RESAMPLING[method]
        original_size = pil.size
        down_h = max(1, round(original_size[1] * target / max(original_size)))
        down_w = max(1, round(original_size[0] * target / max(original_size)))
        pil = pil.resize((down_w, down_h), resample=resample)
        pil = pil.resize(original_size, resample=resample)
        meta["low_resolution"] = {
            "downsample_size": [down_w, down_h],
            "upscale_method": method,
        }

    if rng.random() < config.get("jpeg_probability", 0.0):
        quality = rng.randint(*config["jpeg_quality_range"])
        buffer = BytesIO()
        pil.save(buffer, format="JPEG", quality=quality, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            pil = compressed.convert("RGB").copy()
        meta["jpeg_quality"] = quality

    return np.asarray(pil).copy(), meta, aux

