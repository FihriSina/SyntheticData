#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BlenderProc, import'un dosyanın en başında olmasını zorunlu kılar.
import blenderproc as bproc  # noqa: E402 — bu satır her zaman ilk import olmalı

"""
BlenderProc Sentetik Küp Veri Seti Üreticisi
=============================================
Küp tespiti için YOLO formatında etiketlenmiş sentetik görüntüler üretir.
BlenderProc 2.x API kullanır.
"""

import numpy as np
import random
import os
from collections import deque
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
if BASE_DIR not in sys.path:
    # BlenderProc/Blender betiği geçici bir çalışma bağlamında başlatabilir.
    # Yardımcı modüllerin her zaman generate.py ile aynı klasörden bulunmasını sağlar.
    sys.path.insert(0, BASE_DIR)
import json
import math
import shutil
import traceback
from datetime import datetime

# ================================================================
# CONFIG BLOĞU — Parametreleri buradan düzenleyebilirsiniz
# ================================================================

# — GÖRÜNTÜ ÜRETİMİ —
NUM_IMAGES      = 1000   # Üretilecek toplam görüntü sayısı
RENDER_SAMPLES  = 64     # Cycles render sample sayısı (adaptive sampling açıksa üst sınır)
IMG_RESOLUTION  = 640    # Kare görüntü çözünürlüğü (piksel)
RENDER_PROGRESS_INTERVAL = 50   # Her kaç görüntüde ilerleme raporu

# — RENDER KALİTESİ / HIZ —
RENDER_NOISE_THRESHOLD   = 0.01   # Adaptive sampling eşiği (düşük = kaliteli/yavaş, 0 = kapalı)
USE_DENOISING            = True   # Intel/OptiX denoiser — sample sayısını düşürür
QUALITY_PROFILE          = "balanced"
QUALITY_PROFILES = {
    "fast": {
        "samples": 32, "noise_threshold": 0.030, "max_bounces": 4,
        "diffuse_bounces": 2, "glossy_bounces": 2, "transmission_bounces": 3,
        "volume_bounces": 0, "transparent_bounces": 4,
        "clamp_indirect": 2.0, "filter_glossy": 0.8,
    },
    "balanced": {
        "samples": 64, "noise_threshold": 0.010, "max_bounces": 6,
        "diffuse_bounces": 3, "glossy_bounces": 3, "transmission_bounces": 5,
        "volume_bounces": 1, "transparent_bounces": 6,
        "clamp_indirect": 3.0, "filter_glossy": 0.5,
    },
    "realistic": {
        "samples": 192, "noise_threshold": 0.005, "max_bounces": 8,
        "diffuse_bounces": 4, "glossy_bounces": 4, "transmission_bounces": 8,
        "volume_bounces": 2, "transparent_bounces": 8,
        "clamp_indirect": 4.0, "filter_glossy": 0.25,
    },
}

# — KÜPLER —
MIN_CUBES       = 2      # Görüntü başına minimum küp sayısı
MAX_CUBES       = 12     # Görüntü başına maksimum küp sayısı
MIN_SCALE       = 0.38   # Minimum küp boyutu
MAX_SCALE       = 0.52   # Maksimum küp boyutu
MAX_PLACE_TRIES = 200    # Çakışma önleme için maksimum deneme sayısı
CUBE_COLLISION_MULTIPLIER = 1.5  # Çakışma mesafesinin hesaplanmasında kullanılan çarpan
CUBE_ORIENTATION_WEIGHTS = [0.40, 0.40, 0.20]  # düzgün / küçük eğim / belirgin eğim
CUBE_SMALL_TILT_DEG_RANGE = (2.0, 14.0)
CUBE_STRONG_TILT_DEG_RANGE = (24.0, 68.0)
CUBE_SIDE_LIE_PROBABILITY = 0.55  # Belirgin grupta yaklaşık 90° yüz üstü yatma
STACK_PROBABILITY     = 0.10  # Bir küpün başkasının üstüne yerleşme olasılığı
STACK_XY_JITTER       = 0.08  # Üst üste küplerde kontrollü yatay sapma
CONTACT_GAP           = 0.002 # Z-fighting ve yüzey iç içe geçmesini engeller

# — KAMERA —
MIN_DISTANCE    = 4.5    # Kamera minimum uzaklık (birim)
MAX_DISTANCE    = 10.0   # Kamera maksimum uzaklık (birim)
CAMERA_HEIGHT_MIN = 3.5  # Zeminden minimum kamera yüksekliği
CAMERA_HEIGHT_MAX = 8.5  # Zeminden maksimum kamera yüksekliği
MIN_ELEVATION   = 45     # Minimum kamera yükseklik açısı (derece)
MAX_ELEVATION   = 86     # Maksimum kamera yükseklik açısı (derece)
FOCAL_LENGTH_MIN = 24.0  # Minimum focal length (mm) — geniş açı
FOCAL_LENGTH_MAX = 45.0  # Maksimum focal length (mm) — orta telefoto
CAMERA_INPLANE_ROT_MIN = math.radians(-8.0)  # Minimum kamera roll açısı
CAMERA_INPLANE_ROT_MAX = math.radians(8.0)   # Maksimum kamera roll açısı
FORWARD_VEC_NORM_THRESHOLD = 1e-6  # Forward vector normalization eşik değeri
CAMERA_AIM_JITTER = 0.30  # Kamera odak noktası rastgele sapma mesafesi (m)
USE_DOF           = True  # Derinlik alanı (bokeh) efekti etkin mi
DOF_PROBABILITY   = 0.30  # DoF uygulanacak kare oranı
DOF_FSTOP_MIN     = 2.8   # Minimum f-durağı (küçük = güçlü bokeh)
DOF_FSTOP_MAX     = 8.0   # Maksimum f-durağı

# — ZEMİN —
FLOOR_WIDTH     = 5.5    # Zemin genişliği (m)
FLOOR_HEIGHT    = 3.5    # Zemin yüksekliği (m)
FLOOR_COLOR_MIN = 0.30   # Zemin rengi minimum değer
FLOOR_COLOR_MAX = 0.55   # Zemin rengi maksimum değer
FLOOR_ROUGHNESS_RANGE = (0.45, 0.90)  # Zemin pürüzlülük aralığı
FLOOR_REFLECTION_RANGE = (0.05, 0.28) # Specular/IOR level aralığı
FLOOR_METALLIC_RANGE = (0.0, 0.08)    # Zemin metalik değeri
FLOOR_AREA      = min(FLOOR_WIDTH, FLOOR_HEIGHT) / 2.0  # Küp yerleşim için XY yarı-sınırı (m)
FLOOR_TEXTURE_TYPES   = ["solid", "checker", "grid", "noise", "worn"]
FLOOR_TEXTURE_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

# — KÜPLER RENGİ VE MATERYAL —
CUBE_COLOR_MIN  = 0.15   # Küp rengi minimum değer
CUBE_COLOR_MAX  = 0.95   # Küp rengi maksimum değer
CUBE_MIN_DIST_FROM_FLOOR = 0.35  # Zeminden minimum renk mesafesi
CUBE_COLOR_TRIES = 50    # Uygun renk bulma maksimum deneme sayısı
CUBE_ROUGHNESS_MIN = 0.35  # Küp pürüzlülüğü minimum
CUBE_ROUGHNESS_MAX = 0.85  # Küp pürüzlülüğü maksimum
CUBE_METALLIC   = 0.0    # Küp metalik değeri
SOLID_COLOR_PROBABILITY = 0.30
SOLID_COLORS = {
    "white": (0.92, 0.92, 0.90), "black": (0.025, 0.025, 0.025),
    "gray": (0.42, 0.42, 0.42), "red": (0.72, 0.045, 0.035),
    "blue": (0.035, 0.16, 0.72), "green": (0.035, 0.55, 0.12),
    "yellow": (0.90, 0.72, 0.035), "orange": (0.92, 0.30, 0.025),
    "purple": (0.44, 0.08, 0.62), "pink": (0.90, 0.32, 0.55),
    "brown": (0.30, 0.12, 0.045),
}
SOLID_ROUGHNESS_RANGE = (0.38, 0.72)
SOLID_MICRO_BUMP_RANGE = (0.025, 0.075)
MATERIAL_IOR_RANGE = (1.43, 1.52)
MATERIAL_SPECULAR_IOR_RANGE = (0.28, 0.48)

# — FDM TEXTURE —
USE_FDM_TEXTURES         = True  # False → eski prosedürel sisteme döner
TEXTURE_POOL_SIZE        = 200   # Havuzdaki farklı doku sayısı (1000 görüntü için 100 önerilir)
TEXTURE_POOL_DIR         = os.path.join(BASE_DIR, "fdm_textures")
TEXTURE_SIZE             = 512   # Doku çözünürlüğü (px)
TEXTURE_MASTER_SEED      = 7     # Havuz üretimi için seed
TEXTURE_DIVERSITY_WINDOW = 8     # Son N seçimde kullanılan texture tekrarlanamaz
TEXTURE_UV_SCALE         = 4.0   # Texture tekrar sayısı — yüksek = ince FDM çizgileri
TEXTURE_NORMAL_STRENGTH  = 0.5   # Normal map güç çarpanı (roughness ile çarpılır)
TEXTURE_PATTERN_TYPES = [
    "striped", "grid", "dotted", "geometric", "speckled", "marble",
    "two_tone", "gradient", "worn", "pronounced_layers",
    "filament_wobble", "print_defect",
]
TEXTURE_PATTERN_WEIGHTS = [0.10, 0.08, 0.08, 0.08, 0.10, 0.10,
                           0.08, 0.08, 0.08, 0.08, 0.07, 0.07]

# Pencere tabanlı çeşitlilik için modül seviyesi takipçi
_recent_texture_idx: deque = deque(maxlen=TEXTURE_DIVERSITY_WINDOW)

# — IŞIKLANDIRMA —
LIGHT_TYPES     = ["POINT", "AREA", "SPOT"]  # Işık türleri
MIN_LIGHTS      = 1      # Minimum ışık sayısı
MAX_LIGHTS      = 3      # Maksimum ışık sayısı
LIGHT_X_MIN     = -5.0   # Işık X pozisyonu minimum
LIGHT_X_MAX     = 5.0    # Işık X pozisyonu maksimum
LIGHT_Y_MIN     = -5.0   # Işık Y pozisyonu minimum
LIGHT_Y_MAX     = 5.0    # Işık Y pozisyonu maksimum
LIGHT_Z_MIN     = 2.0    # Işık Z pozisyonu minimum
LIGHT_Z_MAX     = 8.0    # Işık Z pozisyonu maksimum
LIGHT_COLOR_MIN = 0.6    # Işık rengi minimum (nötr)
LIGHT_COLOR_MAX = 1.0    # Işık rengi maksimum (nötr)
LIGHT_ENERGY_MIN = 180.0  # Işık enerjisi minimum
LIGHT_ENERGY_MAX = 1200.0 # Işık enerjisi maksimum
LIGHT_TEMPERATURE_RANGE = (3200, 6800)
AREA_LIGHT_SIZE_RANGE = (1.0, 4.0)
POINT_LIGHT_RADIUS_RANGE = (0.08, 0.45)
SPOT_LIGHT_SIZE_RANGE = (math.radians(28), math.radians(65))
SPOT_BLEND_RANGE = (0.18, 0.55)

# — ARKA PLAN —
HDRI_EXTENSIONS = ["*.hdr", "*.exr"]
IMAGE_BACKGROUND_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]
HDRI_STRENGTH_MIN = 0.5   # Minimum HDRI parlaklığı (karanlık ortam)
HDRI_STRENGTH_MAX = 2.0   # Maksimum HDRI parlaklığı (parlak ortam)
HDRI_TEMPERATURE_RANGE = (4200, 7800)
BACKGROUND_DIVERSITY_WINDOW = 6
IMAGE_BACKGROUND_DISTANCE = 30.0
IMAGE_BACKGROUND_FRAME_MARGIN = 1.08
IMAGE_BACKGROUND_INTERPOLATIONS = ["Linear", "Cubic", "Closest"]
IMAGE_BACKGROUND_INTERPOLATION_WEIGHTS = [0.55, 0.35, 0.10]
IMAGE_BACKGROUND_CROP_JITTER = 0.12

# — ENVIRONMENT / ETİKETLENMEYEN SAHNE NESNELERİ —
ENVIRONMENT_PROBABILITY = 0.72
ENV_OBJECT_COUNT_RANGE = (1, 7)
ENV_BOUNDARY_LINE_PROBABILITY = 0.55
ENV_BARRIER_PROBABILITY = 0.35
ENV_POLE_PROBABILITY = 0.30
ENV_CABLE_PROBABILITY = 0.28
ENV_CYLINDER_PROBABILITY = 0.32
ENV_ROBOT_PART_PROBABILITY = 0.30
ENV_SHADOW_CASTER_PROBABILITY = 0.22
ENV_CONTROLLED_OCCLUDER_PROBABILITY = 0.35
ENV_OBJECT_SIZE_RANGE = (0.06, 0.55)
ENV_HEIGHT_RANGE = (0.08, 1.20)
ENV_CUBE_CLEARANCE = 0.10
ENV_COLORS = [
    (0.06, 0.07, 0.08), (0.18, 0.20, 0.22), (0.45, 0.48, 0.50),
    (0.82, 0.10, 0.04), (0.92, 0.72, 0.04), (0.04, 0.22, 0.68),
]
BOUNDARY_LINE_WIDTH_RANGE = (0.025, 0.075)
BARRIER_LENGTH_RANGE = (0.45, 1.80)
POLE_RADIUS_RANGE = (0.018, 0.055)
CABLE_RADIUS_RANGE = (0.008, 0.025)
SMALL_CYLINDER_RADIUS_RANGE = (0.05, 0.16)

# — GÖRÜNÜRLÜK FİLTRESİ —
MIN_VISIBILITY_RATIO = 0.40  # Küpün kamerada görünmesi gereken minimum oran (%40)
MIN_BBOX_AREA_PX     = 400   # Minimum bbox alanı piksel² (~20×20 px)

# — POST-PROCESSING AUGMENTATION —
POST_AUGMENTATION       = True   # Frame kaydedilirken PIL augmentation uygula
POST_AUG_NOISE_PROB     = 0.70   # Gaussian gürültü uygulama olasılığı
POST_AUG_NOISE_MIN      = 0.5    # Minimum gürültü std. sapması
POST_AUG_NOISE_MAX      = 3.0    # Maksimum gürültü std. sapması
POST_AUG_BLUR_PROB      = 0.30   # Gaussian blur uygulama olasılığı
POST_AUG_BLUR_MIN       = 0.3    # Minimum blur yarıçapı (px)
POST_AUG_BLUR_MAX       = 1.2    # Maksimum blur yarıçapı (px)
POST_AUG_WB_PROB        = 0.50   # Beyaz denge kayması olasılığı
POST_AUG_BC_PROB        = 0.60   # Parlaklık/kontrast değişimi olasılığı
POST_AUG_BRIGHTNESS_MIN = 0.80   # Minimum parlaklık çarpanı
POST_AUG_BRIGHTNESS_MAX = 1.20   # Maksimum parlaklık çarpanı
POST_AUG_CONTRAST_MIN   = 0.85   # Minimum kontrast çarpanı
POST_AUG_CONTRAST_MAX   = 1.15   # Maksimum kontrast çarpanı
POST_AUG_GAMMA_PROB     = 0.35
POST_AUG_GAMMA_RANGE    = (0.88, 1.12)
POST_AUG_TEMPERATURE_RANGE = (4200, 7800)
POST_AUG_MOTION_BLUR_PROB = 0.14
POST_AUG_MOTION_BLUR_RADIUS_RANGE = (1, 2)
POST_AUG_LENS_DISTORTION_PROB = 0.22
POST_AUG_LENS_DISTORTION_RANGE = (-0.018, 0.018)
POST_AUG_EXPOSURE_PROB = 0.22
POST_AUG_EXPOSURE_EV_RANGE = (-0.35, 0.35)
LOW_RES_PROBABILITY = 0.35
LOW_RES_SIZES = [160, 240, 320, 480]
LOW_RES_UPSCALE_METHODS = ["nearest", "bilinear", "bicubic"]
JPEG_COMPRESSION_PROBABILITY = 0.25
JPEG_QUALITY_RANGE = (58, 90)

# — TEKRAR ÜRETİLEBİLİRLİK —
RANDOM_SEED = 42

# — DOSYA YÖNETİMİ —
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
HDRI_DIR        = os.path.join(BASE_DIR, "backgrounds")
IMAGE_BACKGROUND_DIR = os.path.join(BASE_DIR, "textures")
BACKGROUND_ASSET_DIRS = [HDRI_DIR, IMAGE_BACKGROUND_DIR]
YOLO_SUBDIR     = "yolo"
YOLO_IMAGES_SUBDIR = "images"
YOLO_LABELS_SUBDIR = "labels"
YOLO_META_SUBDIR   = "metadata"
COCO_SUBDIR        = "coco"
COCO_IMAGES_SUBDIR = "images"
COCO_ANNOTATIONS_FILE = "coco_annotations.json"

# — DATASET YAML —
DATASET_NC      = 1      # Sınıf sayısı
DATASET_NAMES   = ["cube"]  # Sınıf adları
DATASET_TRAIN_RATIO = 0.8  # Eğitim seti oranı (%80)
DATASET_VAL_RATIO = 0.2    # Doğrulama seti oranı (%20)

# — YOLO FORMATTER —
YOLO_DECIMAL_PLACES = 6  # YOLO formatında ondalak basamak sayısı

# ================================================================

_recent_background_paths = deque(maxlen=BACKGROUND_DIVERSITY_WINDOW)

# ── Komut satırı argümanlarını işle ─────────────────────────────
_test_mode      = "--test"           in sys.argv
_use_gpu        = "--gpu"            in sys.argv
_use_optix      = "--optix"          in sys.argv  # OptiX: NVIDIA RTX/Pascal — en hızlı render
_use_hybrid     = "--hybrid"         in sys.argv  # Cycles CPU+GPU hibrit (NVIDIA için CUDA backend)
_use_eevee      = "--eevee"          in sys.argv  # Eevee: rasterizer, CPU/GPU otomatik, çok hızlı
_regen_textures = "--regen-textures" in sys.argv
_only_textures  = "--gen-textures"   in sys.argv  # Sadece texture üret, render etme

def _get_int_arg(flag, default):
    try:
        return int(sys.argv[sys.argv.index(flag) + 1])
    except (ValueError, IndexError):
        return default


def _get_float_arg(flag, default):
    try:
        return float(sys.argv[sys.argv.index(flag) + 1])
    except (ValueError, IndexError):
        return default


def _get_str_arg(flag, default):
    try:
        return str(sys.argv[sys.argv.index(flag) + 1])
    except (ValueError, IndexError):
        return default


def _get_int_list_arg(flag, default):
    try:
        raw = sys.argv[sys.argv.index(flag) + 1]
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
        return values or list(default)
    except (ValueError, IndexError):
        return list(default)


def _get_int_range_arg(flag, default):
    try:
        idx = sys.argv.index(flag) + 1
        first = sys.argv[idx]
        if "," in first:
            values = [int(value.strip()) for value in first.split(",")]
        else:
            values = [int(first), int(sys.argv[idx + 1])]
        if len(values) != 2:
            raise ValueError
        return min(values), max(values)
    except (ValueError, IndexError):
        return tuple(default)

_worker_id        = _get_int_arg("--worker-id",        0)
_num_workers      = _get_int_arg("--num-workers",       1)
_num_images_arg   = _get_int_arg("--num-images",        0)
_samples_arg      = _get_int_arg("--samples",           0)
_tex_pool_arg     = _get_int_arg("--texture-pool-size", 0)
_quality_profile  = _get_str_arg("--quality-profile", QUALITY_PROFILE).lower()
_seed_arg         = _get_int_arg("--seed", RANDOM_SEED)
_low_res_prob_arg = _get_float_arg("--low-res-probability", LOW_RES_PROBABILITY)
_low_res_sizes_arg = _get_int_list_arg("--low-res-sizes", LOW_RES_SIZES)
_jpeg_prob_arg = _get_float_arg(
    "--jpeg-compression-probability", JPEG_COMPRESSION_PROBABILITY
)
_jpeg_quality_arg = _get_int_range_arg("--jpeg-quality-range", JPEG_QUALITY_RANGE)

if _quality_profile not in QUALITY_PROFILES:
    print(f"UYARI: Bilinmeyen kalite profili '{_quality_profile}', balanced kullanılacak.")
    _quality_profile = "balanced"
QUALITY_PROFILE = _quality_profile
_profile_config = QUALITY_PROFILES[QUALITY_PROFILE]
RENDER_SAMPLES = int(_profile_config["samples"])
RENDER_NOISE_THRESHOLD = float(_profile_config["noise_threshold"])
RANDOM_SEED = _seed_arg
LOW_RES_PROBABILITY = max(0.0, min(1.0, _low_res_prob_arg))
LOW_RES_SIZES = sorted({max(16, min(IMG_RESOLUTION, size)) for size in _low_res_sizes_arg})
JPEG_COMPRESSION_PROBABILITY = max(0.0, min(1.0, _jpeg_prob_arg))
JPEG_QUALITY_RANGE = (
    max(20, min(100, _jpeg_quality_arg[0])),
    max(20, min(100, _jpeg_quality_arg[1])),
)

# CLI argümanları config değerlerini ezer
if _num_images_arg > 0:
    NUM_IMAGES = _num_images_arg
if _samples_arg > 0:
    RENDER_SAMPLES = _samples_arg
if _tex_pool_arg > 0:
    TEXTURE_POOL_SIZE = _tex_pool_arg

if _test_mode:
    NUM_IMAGES        = 10
    RENDER_SAMPLES    = min(RENDER_SAMPLES, 16)
    TEXTURE_POOL_SIZE = 10
    print("TEST MODU: 10 görüntü üretilecek")


# ── Genel yardımcı fonksiyonlar ──────────────────────────────────

def setup_output_dirs(yolo_img_dir, yolo_lbl_dir, yolo_meta_dir):
    for d in [yolo_img_dir, yolo_lbl_dir, yolo_meta_dir]:
        os.makedirs(d, exist_ok=True)


def scan_files(directory, extensions):
    """
    Belirtilen klasördeki belirli uzantılı dosyaları listeler.

    Args:
        directory: Taranacak klasör yolu
        extensions: Uzantı listesi, örn. ['*.hdr', '*.exr']

    Returns:
        list[str]: Tekrarsız, sıralı tam yol listesi
    """
    if not os.path.isdir(directory):
        return []
    suffixes = {extension.replace("*", "").lower() for extension in extensions}
    files = []
    for root, _, names in os.walk(directory):
        for name in names:
            if os.path.splitext(name)[1].lower() in suffixes:
                files.append(os.path.abspath(os.path.join(root, name)))
    return sorted(set(files))


def log_error(frame_idx, message):
    """
    Hata mesajını errors.log dosyasına kaydeder.

    Args:
        frame_idx: Hatalı frame numarası
        message: Hata açıklaması (traceback dahil)
    """
    log_path = os.path.join(OUTPUT_DIR, "errors.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Frame {frame_idx}: {message}\n")


def _configure_cycles_devices(bpy):
    """
    Cycles render cihazlarını seçer.

    Not:
    - OptiX yolu GPU-odaklıdır.
    - CPU+GPU hibrit isteniyorsa CUDA backend tercih edilir.
    """
    use_hybrid = _use_hybrid

    # Hibrit modda OptiX yerine CUDA seçilir (CPU+GPU karması için daha güvenli yol).
    if _use_optix and use_hybrid:
        print("UYARI: --optix + --hybrid birlikte istendi. Hibrit için CUDA backend kullanılacak.")

    if _use_optix and not use_hybrid:
        backend = "OPTIX"
    elif _use_gpu or use_hybrid:
        backend = "CUDA"
    else:
        backend = None

    # BlenderProc API sürümleri arasında parametre farkı var: use_only_cpu geriye uyumlu.
    if backend is not None:
        bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type=backend)

    try:
        bpy.context.scene.cycles.device = "GPU"
    except Exception:
        pass

    active_devices = []
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        try:
            prefs.compute_device_type = backend or prefs.compute_device_type
        except Exception:
            pass
        try:
            prefs.refresh_devices()
        except Exception:
            pass

        for dev in getattr(prefs, "devices", []):
            dev_type = str(getattr(dev, "type", "")).upper()
            if dev_type == "CPU":
                dev.use = bool(use_hybrid)
            else:
                if backend is None:
                    dev.use = True
                else:
                    dev.use = (dev_type == backend)
            if getattr(dev, "use", False):
                active_devices.append(f"{getattr(dev, 'name', 'unknown')}[{dev_type}]")
    except Exception as dev_err:
        print(f"Cihaz tercihleri detaylandırılamadı: {dev_err}")

    if backend is None:
        print("Cycles cihaz modu: varsayılan (uygun GPU backend otomatik seçilir).")
    elif use_hybrid:
        print(f"Cycles cihaz modu: {backend} + CPU hibrit")
    else:
        print(f"Cycles cihaz modu: {backend} (GPU-only)")

    if active_devices:
        print("Aktif Cycles cihazları: " + ", ".join(active_devices))
    else:
        print("Aktif Cycles cihazları raporlanamadı.")

    return backend


def _safe_setattr(obj, name, value):
    try:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return True
    except Exception:
        pass
    return False


def _configure_color_management(scene):
    """AgX varsa kullanır; eski Blender sürümlerinde Filmic/Standard'a düşer."""
    selected = None
    for transform in ("AgX", "Filmic", "Standard"):
        try:
            scene.view_settings.view_transform = transform
            selected = transform
            break
        except Exception:
            continue
    if selected == "AgX":
        for look in ("AgX - Medium High Contrast", "AgX - Medium Low Contrast", "Medium High Contrast"):
            try:
                scene.view_settings.look = look
                break
            except Exception:
                continue
    _safe_setattr(scene.view_settings, "exposure", 0.0)
    _safe_setattr(scene.view_settings, "gamma", 1.0)
    return selected or "unknown"


def _configure_cycles_quality(bpy, backend):
    """Seçili profile göre Cycles kalite, bounce ve noise ayarlarını uygular."""
    scene = bpy.context.scene
    cycles = scene.cycles
    profile = QUALITY_PROFILES[QUALITY_PROFILE]
    bproc.renderer.set_max_amount_of_samples(RENDER_SAMPLES)
    _safe_setattr(cycles, "use_adaptive_sampling", RENDER_NOISE_THRESHOLD > 0)
    _safe_setattr(cycles, "adaptive_threshold", RENDER_NOISE_THRESHOLD)
    _safe_setattr(cycles, "adaptive_min_samples", max(4, RENDER_SAMPLES // 16))
    _safe_setattr(cycles, "use_denoising", USE_DENOISING)
    _safe_setattr(cycles, "use_preview_denoising", USE_DENOISING)

    for key in (
        "max_bounces", "diffuse_bounces", "glossy_bounces",
        "transmission_bounces", "volume_bounces", "transparent_bounces",
        "clamp_indirect", "filter_glossy",
    ):
        _safe_setattr(cycles, key, profile[key])
    _safe_setattr(cycles, "clamp_direct", 0.0)
    _safe_setattr(cycles, "use_fast_gi", True)
    _safe_setattr(cycles, "ao_bounces", 3)
    _safe_setattr(cycles, "ao_bounces_render", 3)
    _safe_setattr(cycles, "sample_clamp_indirect", profile["clamp_indirect"])

    if USE_DENOISING:
        candidates = ["OPTIX", "OPENIMAGEDENOISE"] if backend == "OPTIX" else ["OPENIMAGEDENOISE", "OPTIX"]
        for denoiser in candidates:
            try:
                cycles.denoiser = denoiser
                break
            except Exception:
                continue
    color_management = _configure_color_management(scene)
    return {
        "profile": QUALITY_PROFILE,
        "samples": RENDER_SAMPLES,
        "noise_threshold": RENDER_NOISE_THRESHOLD,
        "denoising": USE_DENOISING,
        "denoiser": getattr(cycles, "denoiser", None),
        "color_management": color_management,
        "bounces": {key: profile[key] for key in (
            "max_bounces", "diffuse_bounces", "glossy_bounces",
            "transmission_bounces", "volume_bounces", "transparent_bounces",
        )},
        "clamp_indirect": profile["clamp_indirect"],
        "filter_glossy": profile["filter_glossy"],
    }


# ── Sahne arka plan ve ışık fonksiyonları ────────────────────────

def _kelvin_to_rgb(kelvin):
    """Yaklaşık renk sıcaklığını normalize RGB'ye dönüştürür."""
    temperature = max(1000.0, min(40000.0, float(kelvin))) / 100.0
    if temperature <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temperature) - 161.1195681661
        blue = 0.0 if temperature <= 19.0 else (
            138.5177312231 * math.log(temperature - 10.0) - 305.0447927307
        )
    else:
        red = 329.698727446 * ((temperature - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temperature - 60.0) ** -0.0755148492)
        blue = 255.0
    return [max(0.0, min(1.0, value / 255.0)) for value in (red, green, blue)]


def _set_node_input(node, names, value):
    """Blender 3.x/4.x Principled giriş adları için güvenli atama."""
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return True
    return False


def select_background_source(hdri_files, image_files):
    """Tüm EXR/HDR ve JPG/PNG dosyalarının ortak havuzundan bir dosya seçer."""
    sources = sorted({
        ("hdri", os.path.abspath(path)) for path in hdri_files
    } | {
        ("image", os.path.abspath(path)) for path in image_files
    }, key=lambda item: (item[1], item[0]))
    if not sources:
        return {"source_type": "random_color", "path": None, "file_name": None}
    available = [
        source for source in sources
        if source[1] not in _recent_background_paths
    ]
    source_type, selected_path = random.choice(available or sources)
    _recent_background_paths.append(selected_path)
    return {
        "source_type": source_type,
        "path": selected_path,
        "file_name": os.path.basename(selected_path),
    }


def _relative_asset_path(path):
    if not path:
        return None
    try:
        return os.path.relpath(path, BASE_DIR).replace(os.sep, "/")
    except ValueError:
        return os.path.basename(path)


def _apply_world_hdri(hdri_path, rotation, strength, kelvin):
    """World node ağını sıfırlayıp yalnızca seçilen HDRI dosyasını bağlar."""
    import bpy  # type: ignore

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("SyntheticWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, rotation)
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(hdri_path, check_existing=True)
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = strength
    output = nodes.new("ShaderNodeOutputWorld")

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    try:
        tint = nodes.new("ShaderNodeMixRGB")
        tint.blend_type = "MULTIPLY"
        tint.inputs[0].default_value = 1.0
        tint.inputs[2].default_value = _kelvin_to_rgb(kelvin) + [1.0]
        links.new(environment.outputs["Color"], tint.inputs[1])
        links.new(tint.outputs["Color"], background.inputs["Color"])
    except Exception:
        links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

    expected = os.path.normcase(os.path.abspath(hdri_path))
    actual = os.path.normcase(os.path.abspath(bpy.path.abspath(environment.image.filepath)))
    if actual != expected:
        raise RuntimeError(
            f"HDRI node doğrulaması başarısız: beklenen={expected}, bağlı={actual}"
        )
    return actual


def _apply_neutral_world(color):
    """Önceki kareden kalan HDRI node'larını kaldırıp düz world rengi uygular."""
    import bpy  # type: ignore

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("SyntheticWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = color
    background.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(background.outputs["Background"], output.inputs["Surface"])


def setup_background(background_source):
    """Karede yalnızca seçilen tek arka plan kaynağını etkinleştirir."""
    rotation = random.uniform(0.0, 2.0 * math.pi)
    strength = random.uniform(HDRI_STRENGTH_MIN, HDRI_STRENGTH_MAX)
    kelvin = random.uniform(*HDRI_TEMPERATURE_RANGE)
    source_type = background_source.get("source_type", "random_color")
    source_path = background_source.get("path")
    if source_type == "hdri" and source_path:
        applied_path = _apply_world_hdri(source_path, rotation, strength, kelvin)
        return {
            "source_type": "hdri", "name": os.path.basename(source_path),
            "relative_path": _relative_asset_path(source_path),
            "applied_name": os.path.basename(applied_path),
            "applied_relative_path": _relative_asset_path(applied_path),
            "rotation_deg": round(math.degrees(rotation), 2),
            "strength": round(strength, 3), "temperature_k": round(kelvin),
        }

    # JPG/PNG karelerinde herhangi bir EXR/HDR yüklenmez. Nötr world yalnızca
    # fiziksel sahne ışıklarının kaçırdığı ışınlar için güvenli fallback'tir.
    color = [random.uniform(0.16, 0.34) for _ in range(3)] + [1.0]
    _apply_neutral_world(color)
    if source_type == "image" and source_path:
        return {
            "source_type": "image", "name": os.path.basename(source_path),
            "relative_path": _relative_asset_path(source_path),
            "world_color": [round(value, 3) for value in color[:3]],
            "rotation_deg": None, "strength": None, "temperature_k": None,
        }

    return {
        "source_type": "random_color", "name": "random_color_background",
        "relative_path": None,
        "world_color": [round(value, 3) for value in color[:3]],
        "rotation_deg": None, "strength": None, "temperature_k": None,
    }


def direction_to_euler(direction):
    """Bir Blender nesnesinin yerel -Z eksenini verilen yöne çevirir."""
    import mathutils  # type: ignore
    vector = mathutils.Vector(direction)
    if vector.length < FORWARD_VEC_NORM_THRESHOLD:
        return mathutils.Euler((0.0, 0.0, 0.0))
    return vector.to_track_quat("-Z", "Y").to_euler()


def create_scene_lights():
    """
    1-3 arası rastgele tipte ve konumda ışık oluşturur.

    Returns:
        list[bproc.types.Light]: Oluşturulan ışık nesneleri
    """
    lights = []
    light_meta = []
    for _ in range(random.randint(MIN_LIGHTS, MAX_LIGHTS)):
        light = bproc.types.Light()
        light_type = random.choice(LIGHT_TYPES)
        location = [
            random.uniform(LIGHT_X_MIN, LIGHT_X_MAX),
            random.uniform(LIGHT_Y_MIN, LIGHT_Y_MAX),
            random.uniform(LIGHT_Z_MIN, LIGHT_Z_MAX),
        ]
        kelvin = random.uniform(*LIGHT_TEMPERATURE_RANGE)
        base_color = _kelvin_to_rgb(kelvin)
        color = [
            max(0.0, min(1.0, component * random.uniform(0.94, 1.06)))
            for component in base_color
        ]
        energy = random.uniform(LIGHT_ENERGY_MIN, LIGHT_ENERGY_MAX)
        light.set_type(light_type)
        light.set_location(location)
        light.set_color(color)
        light.set_energy(energy)
        softness = None
        try:
            data = light.blender_obj.data
            data.use_shadow = True
            if light_type == "AREA":
                softness = random.uniform(*AREA_LIGHT_SIZE_RANGE)
                data.shape = random.choice(["DISK", "RECTANGLE"])
                data.size = softness
            elif light_type == "POINT":
                softness = random.uniform(*POINT_LIGHT_RADIUS_RANGE)
                data.shadow_soft_size = softness
            elif light_type == "SPOT":
                softness = random.uniform(*POINT_LIGHT_RADIUS_RANGE)
                data.shadow_soft_size = softness
                data.spot_size = random.uniform(*SPOT_LIGHT_SIZE_RANGE)
                data.spot_blend = random.uniform(*SPOT_BLEND_RANGE)
                direction = np.array([0.0, 0.0, 0.0]) - np.asarray(location)
                light.blender_obj.rotation_euler = direction_to_euler(direction)
            if hasattr(data, "use_contact_shadow"):
                data.use_contact_shadow = True
        except Exception:
            pass
        lights.append(light)
        light_meta.append({
            "type": light_type,
            "location": [round(value, 3) for value in location],
            "temperature_k": round(kelvin),
            "color": [round(value, 3) for value in color],
            "energy": round(energy, 2),
            "softness": round(softness, 3) if softness is not None else None,
        })
    return lights, light_meta


# ── Materyal fonksiyonları ───────────────────────────────────────

def generate_texture_pool(pool_size=TEXTURE_POOL_SIZE, size=TEXTURE_SIZE, master_seed=TEXTURE_MASTER_SEED):
    """
    texture.py kullanarak FDM albedo+normal çiftlerini önceden üretir.
    İkinci çalışmada disk cache'den yükler.

    Returns:
        list[dict]: [{albedo, normal, roughness, metallic, material, color}, ...]
    """
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    from texture import generate_variant, MATERIAL_PROFILES

    os.makedirs(TEXTURE_POOL_DIR, exist_ok=True)
    meta_path = os.path.join(TEXTURE_POOL_DIR, "pool_meta.json")

    refresh_existing = False
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            pool = json.load(f)
        if len(pool) >= pool_size and all(
            item.get("pattern_type") and item.get("seed") is not None
            for item in pool[:pool_size]
        ):
            print(f"FDM texture havuzu cache'den yüklendi: {len(pool)} doku → {TEXTURE_POOL_DIR}")
            return pool
        refresh_existing = True

    rng = np.random.default_rng(master_seed)

    all_variants = []
    for mat_name, prof in MATERIAL_PROFILES.items():
        for color_name, rgb in prof["colors"]:
            all_variants.append((mat_name, color_name, rgb, prof))

    selected = []
    while len(selected) < pool_size:
        order = rng.permutation(len(all_variants))
        for idx in order:
            selected.append(all_variants[idx])
            if len(selected) >= pool_size:
                break

    pool = []
    print(f"FDM texture havuzu üretiliyor: {pool_size} varyasyon → {TEXTURE_POOL_DIR}")
    for i, (mat, col, rgb, prof) in enumerate(selected):
        seed = int(rng.integers(0, 999_999))
        vid  = f"{i:03d}_{mat}_{col}"
        pattern_type = str(rng.choice(TEXTURE_PATTERN_TYPES, p=TEXTURE_PATTERN_WEIGHTS))
        solid_name = str(rng.choice(list(SOLID_COLORS)))
        secondary_rgb = tuple(int(round(channel * 255)) for channel in SOLID_COLORS[solid_name])

        albedo_path = os.path.join(TEXTURE_POOL_DIR, f"{vid}_albedo.png")
        normal_path = os.path.join(TEXTURE_POOL_DIR, f"{vid}_normal.png")

        if refresh_existing or not os.path.isfile(albedo_path) or not os.path.isfile(normal_path):
            albedo_img, normal_img, variant_meta = generate_variant(
                mat, col, rgb, prof, seed, size,
                pattern_type=pattern_type, secondary_rgb=secondary_rgb,
            )
            albedo_img.save(albedo_path)
            normal_img.save(normal_path)
        else:
            variant_meta = {
                "pattern_type": pattern_type,
                "secondary_rgb": list(secondary_rgb),
            }

        pool.append({
            "albedo":    albedo_path,
            "normal":    normal_path,
            "roughness": prof["roughness"],
            "metallic":  prof.get("metallic", 0.0),
            "material":  mat,
            "color":     col,
            "base_rgb": list(rgb),
            "secondary_rgb": variant_meta["secondary_rgb"],
            "pattern_type": variant_meta["pattern_type"],
            "seed": seed,
        })

        if (i + 1) % 10 == 0 or (i + 1) == pool_size:
            print(f"  {i + 1}/{pool_size} tamamlandı")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)

    print(f"Texture havuzu hazır: {len(pool)} doku")
    return pool


def _pick_diverse_texture(pool):
    """
    Pencere tabanlı çeşitlilik: son TEXTURE_DIVERSITY_WINDOW seçimde
    kullanılan index tekrar seçilemez.
    """
    available = [i for i in range(len(pool)) if i not in _recent_texture_idx]
    if not available:
        available = list(range(len(pool)))
    idx = random.choice(available)
    _recent_texture_idx.append(idx)
    return pool[idx]


def make_fdm_texture_material(entry):
    """
    Albedo + normal map içeren Principled BSDF materyali oluşturur.
    entry = {albedo, normal, roughness, metallic, material, color}
    """
    import bpy  # type: ignore

    mat_name = f"fdm_{entry['material']}_{entry['color']}_{random.randint(0, 999_999)}"
    mat = bproc.material.create(mat_name)

    bpy_mat = mat.blender_obj
    bpy_mat.use_nodes = True
    nodes = bpy_mat.node_tree.nodes
    links = bpy_mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = entry["roughness"]
    bsdf.inputs["Metallic"].default_value  = entry.get("metallic", 0.0)
    _set_node_input(bsdf, ("IOR",), random.uniform(*MATERIAL_IOR_RANGE))
    _set_node_input(
        bsdf, ("Specular IOR Level", "Specular"),
        random.uniform(*MATERIAL_SPECULAR_IOR_RANGE),
    )

    out_node = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping   = nodes.new("ShaderNodeMapping")
    s = TEXTURE_UV_SCALE
    mapping.inputs["Scale"].default_value = (s, s, s)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    img_albedo = bpy.data.images.load(entry["albedo"], check_existing=True)
    tex_albedo = nodes.new("ShaderNodeTexImage")
    tex_albedo.image = img_albedo
    links.new(mapping.outputs["Vector"],   tex_albedo.inputs["Vector"])
    links.new(tex_albedo.outputs["Color"], bsdf.inputs["Base Color"])

    img_normal = bpy.data.images.load(entry["normal"], check_existing=True)
    img_normal.colorspace_settings.name = "Non-Color"
    tex_normal = nodes.new("ShaderNodeTexImage")
    tex_normal.image = img_normal
    links.new(mapping.outputs["Vector"],  tex_normal.inputs["Vector"])

    nrm_node = nodes.new("ShaderNodeNormalMap")
    nrm_node.inputs["Strength"].default_value = entry["roughness"] * TEXTURE_NORMAL_STRENGTH
    links.new(tex_normal.outputs["Color"], nrm_node.inputs["Color"])
    links.new(nrm_node.outputs["Normal"],  bsdf.inputs["Normal"])

    return mat


def make_solid_color_material(color_name, rgb):
    """Hafif mikro pürüzlü, dokusuz görünümlü FDM/plastik materyal."""
    mat = bproc.material.create(f"solid_{color_name}_{random.randint(0, 999_999)}")
    bpy_mat = mat.blender_obj
    bpy_mat.use_nodes = True
    nodes = bpy_mat.node_tree.nodes
    links = bpy_mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = list(rgb) + [1.0]
    roughness = random.uniform(*SOLID_ROUGHNESS_RANGE)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    _set_node_input(bsdf, ("IOR",), random.uniform(*MATERIAL_IOR_RANGE))
    _set_node_input(
        bsdf, ("Specular IOR Level", "Specular"),
        random.uniform(*MATERIAL_SPECULAR_IOR_RANGE),
    )

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = random.uniform(90.0, 180.0)
    noise.inputs["Detail"].default_value = random.uniform(2.0, 5.0)
    noise.inputs["Roughness"].default_value = 0.55
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = random.uniform(*SOLID_MICRO_BUMP_RANGE)
    bump.inputs["Distance"].default_value = 0.035
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    out_node = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
    return mat, {
        "kind": "solid", "color_name": color_name,
        "primary_color": [round(value, 4) for value in rgb],
        "secondary_color": None, "pattern_type": "solid",
        "roughness": round(roughness, 4), "metallic": 0.0,
        "micro_surface": True,
    }


# ── Arka plan ve zemin materyal fonksiyonları ───────────────────

def _build_image_backdrop_material(image_path):
    """JPG/JPEG/PNG dosyasını kamera arkasındaki emissive düzleme bağlar."""
    import bpy  # type: ignore

    mat = bproc.material.create(f"backdrop_image_{random.randint(0, 999_999)}")
    bpy_mat = mat.blender_obj
    bpy_mat.use_nodes = True
    nodes = bpy_mat.node_tree.nodes
    links = bpy_mat.node_tree.links
    nodes.clear()

    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    image = bpy.data.images.load(image_path, check_existing=True)
    try:
        image.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    interpolation = random.choices(
        IMAGE_BACKGROUND_INTERPOLATIONS,
        weights=IMAGE_BACKGROUND_INTERPOLATION_WEIGHTS,
        k=1,
    )[0]
    try:
        texture.interpolation = interpolation
    except Exception:
        interpolation = "Linear"
        texture.interpolation = interpolation
    texture.extension = "EXTEND"

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    image_width = max(1, int(image.size[0]))
    image_height = max(1, int(image.size[1]))
    image_aspect = image_width / image_height
    output_aspect = 1.0
    scale_x = scale_y = 1.0
    offset_x = offset_y = 0.0
    if image_aspect > output_aspect:
        scale_x = output_aspect / image_aspect
        slack = 1.0 - scale_x
        offset_x = slack / 2.0 + random.uniform(
            -slack * IMAGE_BACKGROUND_CROP_JITTER,
            slack * IMAGE_BACKGROUND_CROP_JITTER,
        )
    elif image_aspect < output_aspect:
        scale_y = image_aspect / output_aspect
        slack = 1.0 - scale_y
        offset_y = slack / 2.0 + random.uniform(
            -slack * IMAGE_BACKGROUND_CROP_JITTER,
            slack * IMAGE_BACKGROUND_CROP_JITTER,
        )
    mapping.inputs["Scale"].default_value = (scale_x, scale_y, 1.0)
    mapping.inputs["Location"].default_value = (offset_x, offset_y, 0.0)
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], emission.inputs["Color"])

    return mat, {
        "background_file": os.path.basename(image_path),
        "background_relative_path": _relative_asset_path(image_path),
        "source_resolution": [image_width, image_height],
        "interpolation": interpolation.lower(),
        "mapping_scale": [round(scale_x, 5), round(scale_y, 5)],
        "mapping_offset": [round(offset_x, 5), round(offset_y, 5)],
        "display_mode": "camera_facing_plane",
    }

def _build_floor_material(floor_color, floor_type, roughness, metallic, reflection):
    """
    Verilen zemin tipine göre Principled BSDF materyali oluşturur.

    Args:
        floor_color: [r, g, b] temel renk
        floor_type:  "solid" | "checker" | "grid" | "noise" | "worn"

    Returns:
        bproc.types.Material
    """
    mat_name = f"floor_{floor_type}_{random.randint(0, 9999)}"

    if floor_type == "solid":
        mat = bproc.material.create(mat_name)
        mat.set_principled_shader_value("Base Color", floor_color + [1.0])
        mat.set_principled_shader_value("Roughness", roughness)
        mat.set_principled_shader_value("Metallic", metallic)
        for key in ("Specular IOR Level", "Specular"):
            try:
                mat.set_principled_shader_value(key, reflection)
                break
            except Exception:
                pass
        return mat

    # Node tabanlı materyaller
    mat     = bproc.material.create(mat_name)
    bpy_mat = mat.blender_obj
    bpy_mat.use_nodes = True
    nodes = bpy_mat.node_tree.nodes
    links = bpy_mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value  = metallic
    _set_node_input(bsdf, ("Specular IOR Level", "Specular"), reflection)
    _set_node_input(bsdf, ("IOR",), 1.48)

    out_node = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    tex_coord = nodes.new("ShaderNodeTexCoord")

    if floor_type == "checker":
        checker = nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = random.uniform(4.0, 8.0)
        c2 = [min(1.0, c + random.uniform(0.10, 0.25)) for c in floor_color]
        checker.inputs["Color1"].default_value = floor_color + [1.0]
        checker.inputs["Color2"].default_value = c2 + [1.0]
        links.new(tex_coord.outputs["Generated"], checker.inputs["Vector"])
        links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])

    elif floor_type == "grid":
        # Checker + mapping ile izgara görünümü
        mapping = nodes.new("ShaderNodeMapping")
        g = random.uniform(5.0, 10.0)
        mapping.inputs["Scale"].default_value = (g, g, 1.0)
        checker = nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = 1.0
        c_line = [max(0.0, c - 0.20) for c in floor_color]
        checker.inputs["Color1"].default_value = floor_color + [1.0]
        checker.inputs["Color2"].default_value = c_line + [1.0]
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"],      checker.inputs["Vector"])
        links.new(checker.outputs["Color"],       bsdf.inputs["Base Color"])

    elif floor_type in ("noise", "worn"):
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value    = random.uniform(2.0, 6.0)
        noise.inputs["Detail"].default_value   = 5.0
        noise.inputs["Roughness"].default_value = 0.65
        if floor_type == "worn":
            noise.inputs["Distortion"].default_value = random.uniform(0.5, 2.0)

        ramp  = nodes.new("ShaderNodeValToRGB")
        delta = 0.15 if floor_type == "noise" else 0.25
        dark  = [max(0.0, c - delta) for c in floor_color]
        ramp.color_ramp.elements[0].color = dark + [1.0]
        ramp.color_ramp.elements[1].color = floor_color + [1.0]
        links.new(tex_coord.outputs["Generated"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"],           ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"],          bsdf.inputs["Base Color"])

    return mat


def create_image_backdrop(background_source):
    """Seçilen raster görseli kamerayı tamamen kaplayan düzleme yerleştirir."""
    if (background_source or {}).get("source_type") != "image":
        return None, {}
    image_path = background_source.get("path")
    if not image_path:
        return None, {}

    import bpy  # type: ignore
    import mathutils  # type: ignore

    camera = bpy.context.scene.camera
    if camera is None:
        raise RuntimeError("Raster arka plan oluşturmak için aktif kamera bulunamadı.")
    distance = max(float(IMAGE_BACKGROUND_DISTANCE), float(camera.data.clip_start) + 1.0)
    forward = camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
    location = camera.matrix_world.translation + forward * distance
    half_width = distance * math.tan(float(camera.data.angle_x) / 2.0)
    half_height = distance * math.tan(float(camera.data.angle_y) / 2.0)

    backdrop = bproc.object.create_primitive("PLANE")
    backdrop.set_scale([
        half_width * IMAGE_BACKGROUND_FRAME_MARGIN,
        half_height * IMAGE_BACKGROUND_FRAME_MARGIN,
        1.0,
    ])
    backdrop.set_location(list(location))
    backdrop.set_rotation_euler((-forward).to_track_quat("Z", "Y").to_euler())
    material, image_meta = _build_image_backdrop_material(image_path)
    backdrop.replace_materials(material)
    backdrop.blender_obj.name = "env_image_background"
    backdrop.blender_obj.pass_index = 0
    backdrop.set_cp("category_id", 0)
    backdrop.set_cp("is_cube", False)
    backdrop.set_cp("environment_type", "image_background")
    for attribute in (
        "visible_shadow", "visible_diffuse", "visible_glossy",
        "visible_transmission", "visible_volume_scatter",
    ):
        _safe_setattr(backdrop.blender_obj, attribute, False)
    return backdrop, {
        **image_meta,
        "distance": round(distance, 3),
        "frame_margin": IMAGE_BACKGROUND_FRAME_MARGIN,
    }


def create_floor(background_source=None):
    """Arka plan türü için görünmez temas düzlemi veya güvenli fallback zemin."""
    source_type = (background_source or {}).get("source_type", "random_color")
    floor_color = [
        random.uniform(FLOOR_COLOR_MIN, FLOOR_COLOR_MAX),
        random.uniform(FLOOR_COLOR_MIN, FLOOR_COLOR_MAX),
        random.uniform(FLOOR_COLOR_MIN, FLOOR_COLOR_MAX),
    ]

    if source_type in {"hdri", "image"}:
        # Kaynaklı karelerde görünür zemin yoktur. Cycles destekliyorsa yalnızca
        # temas gölgesini yakalayan şeffaf bir düzlem kullanılır.
        floor_type = f"{source_type}_background_only"
        if _use_eevee:
            return None, floor_color, {
                "type": floor_type, "visible": False,
                "shadow_catcher": False,
            }
        floor = bproc.object.create_primitive("PLANE")
        floor.set_scale([
            FLOOR_WIDTH / 2,
            FLOOR_HEIGHT / 2,
            1,
        ])
        floor.set_location([0, 0, 0])
        try:
            floor.blender_obj.is_shadow_catcher = True
            return floor, floor_color, {
                "type": f"{source_type}_shadow_catcher", "visible": False,
                "shadow_catcher": True,
            }
        except Exception:
            floor.delete()
            return None, floor_color, {
                "type": floor_type, "visible": False,
                "shadow_catcher": False,
            }

    floor = bproc.object.create_primitive("PLANE")
    floor.set_scale([
        FLOOR_WIDTH / 2,
        FLOOR_HEIGHT / 2,
        1,
    ])
    floor.set_location([0, 0, 0])

    floor_type = random.choices(FLOOR_TEXTURE_TYPES, FLOOR_TEXTURE_WEIGHTS)[0]
    roughness = random.uniform(*FLOOR_ROUGHNESS_RANGE)
    metallic = random.uniform(*FLOOR_METALLIC_RANGE)
    reflection = random.uniform(*FLOOR_REFLECTION_RANGE)
    mat = _build_floor_material(
        floor_color, floor_type, roughness, metallic, reflection
    )
    floor.replace_materials(mat)

    return floor, floor_color, {
        "type": floor_type,
        "color": [round(value, 4) for value in floor_color],
        "roughness": round(roughness, 4),
        "metallic": round(metallic, 4),
        "reflection": round(reflection, 4),
    }


# ── Küp yerleşim fonksiyonları ───────────────────────────────────

def color_distance(c1, c2):
    """İki RGB renk arasındaki öklidyen mesafe."""
    return ((c1[0] - c2[0]) ** 2 +
            (c1[1] - c2[1]) ** 2 +
            (c1[2] - c2[2]) ** 2) ** 0.5


def random_color_far_from_floor(floor_color, min_dist=None, max_tries=None):
    """Zeminden yeterince farklı bir küp rengi üretir."""
    if min_dist is None:
        min_dist = CUBE_MIN_DIST_FROM_FLOOR
    if max_tries is None:
        max_tries = CUBE_COLOR_TRIES

    for _ in range(max_tries):
        candidate = [
            random.uniform(CUBE_COLOR_MIN, CUBE_COLOR_MAX),
            random.uniform(CUBE_COLOR_MIN, CUBE_COLOR_MAX),
            random.uniform(CUBE_COLOR_MIN, CUBE_COLOR_MAX),
        ]
        if color_distance(candidate, floor_color) >= min_dist:
            return candidate

    avg = sum(floor_color) / 3.0
    if avg > 0.5:
        return [CUBE_COLOR_MIN, CUBE_COLOR_MIN, CUBE_COLOR_MIN]
    else:
        return [CUBE_COLOR_MAX, CUBE_COLOR_MAX, CUBE_COLOR_MAX]


def find_non_overlapping_position(placed_positions, placed_radii, new_radius, area=None):
    """
    Mevcut küplerle çakışmayan rastgele bir (x, y) konumu arar.
    placed_radii: her yerleştirilmiş küpün etkin XY yarıçapı (tilt dahil)
    new_radius:   yeni küpün etkin XY yarıçapı
    Maksimum MAX_PLACE_TRIES denemeden sonra bulunamazsa None döner.
    """
    if area is None:
        area = FLOOR_AREA

    for _ in range(MAX_PLACE_TRIES):
        x = random.uniform(-area, area)
        y = random.uniform(-area, area)

        collision = False
        for (px, py, _), pr in zip(placed_positions, placed_radii):
            if math.hypot(x - px, y - py) < new_radius + pr:
                collision = True
                break

        if not collision:
            return x, y

    return None


def _rotation_matrix_xyz(rotation):
    """Blender XYZ Euler açılarından 3x3 dönüş matrisi üretir."""
    rx, ry, rz = rotation
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rot_z @ rot_y @ rot_x


def _sample_signed_angle(degree_range):
    angle = random.uniform(*degree_range)
    return math.radians(angle if random.random() < 0.5 else -angle)


def sample_cube_orientation():
    """%40 düzgün, %40 küçük eğimli, %20 belirgin/yatık dönüş üretir."""
    mode = random.choices(
        ["upright", "small_tilt", "strong_tilt"], CUBE_ORIENTATION_WEIGHTS
    )[0]
    rot_x = rot_y = 0.0
    if mode == "small_tilt":
        rot_x = _sample_signed_angle(CUBE_SMALL_TILT_DEG_RANGE)
        rot_y = _sample_signed_angle(CUBE_SMALL_TILT_DEG_RANGE)
    elif mode == "strong_tilt":
        if random.random() < CUBE_SIDE_LIE_PROBABILITY:
            side_angle = math.radians(random.choice([-90.0, 90.0]) + random.uniform(-7.0, 7.0))
            if random.random() < 0.5:
                rot_x = side_angle
                rot_y = math.radians(random.uniform(-10.0, 10.0))
            else:
                rot_y = side_angle
                rot_x = math.radians(random.uniform(-10.0, 10.0))
        else:
            rot_x = _sample_signed_angle(CUBE_STRONG_TILT_DEG_RANGE)
            rot_y = _sample_signed_angle(CUBE_STRONG_TILT_DEG_RANGE)
    return (rot_x, rot_y, math.radians(random.uniform(0.0, 360.0))), mode


def _rotated_half_extents(scale, rotation):
    matrix = np.abs(_rotation_matrix_xyz(rotation))
    return matrix @ np.array([scale, scale, scale], dtype=float)


def _place_cube_on_support(cube, x, y, scale, rotation, support_z=0.0):
    """Döndürülmüş küpün en alt noktasını destek düzlemine tam oturtur."""
    half_extents = _rotated_half_extents(scale, rotation)
    z = float(support_z + half_extents[2] + CONTACT_GAP)
    cube.set_location([x, y, z])
    return z, half_extents


def create_cubes(floor_color, fdm_pool=None):
    """
    Sahnede rastgele sayı ve özellikte küp oluşturur.
    X/Y tilt, stacking ve per-cube metadata desteği içerir.

    Returns:
        tuple: (cubes_list, cube_meta_list)
    """
    num_cubes = random.randint(MIN_CUBES, MAX_CUBES)
    cubes     = []
    positions = []
    scales    = []
    radii     = []
    cube_metas = []

    for _ in range(num_cubes):
        scale  = random.uniform(MIN_SCALE, MAX_SCALE)

        rotation, orientation_mode = sample_cube_orientation()
        tilt_x, tilt_y, z_rot = rotation
        half_extents = _rotated_half_extents(scale, rotation)
        effective_radius = float(math.hypot(half_extents[0], half_extents[1]))

        pos_xy = find_non_overlapping_position(positions, radii, effective_radius)

        if pos_xy is None:
            continue

        x, y = pos_xy
        cube = bproc.object.create_primitive("CUBE")
        cube.set_scale([scale, scale, scale])
        cube.set_rotation_euler(rotation)
        z, half_extents = _place_cube_on_support(
            cube, x, y, scale, rotation, support_z=0.0
        )

        if random.random() < SOLID_COLOR_PROBABILITY:
            color_name = random.choice(list(SOLID_COLORS))
            mat, material_meta = make_solid_color_material(
                color_name, SOLID_COLORS[color_name]
            )
            tex_name = f"solid_{color_name}"
        elif fdm_pool and USE_FDM_TEXTURES:
            entry = _pick_diverse_texture(fdm_pool)
            mat   = make_fdm_texture_material(entry)
            tex_name = f"{entry['material']}_{entry['color']}_{entry.get('pattern_type', 'fdm')}"
            material_meta = {
                "kind": "fdm_texture",
                "material": entry["material"],
                "color_name": entry["color"],
                "primary_color": entry.get("base_rgb"),
                "secondary_color": entry.get("secondary_rgb"),
                "pattern_type": entry.get("pattern_type", "fdm_layers"),
                "roughness": entry["roughness"],
                "metallic": entry.get("metallic", 0.0),
                "texture_seed": entry.get("seed"),
            }
        else:
            cube_rgb = random_color_far_from_floor(floor_color)
            mat = bproc.material.create("cube_contrast_material")
            mat.set_principled_shader_value("Base Color", cube_rgb + [1.0])
            roughness = random.uniform(CUBE_ROUGHNESS_MIN, CUBE_ROUGHNESS_MAX)
            mat.set_principled_shader_value("Roughness", roughness)
            mat.set_principled_shader_value("Metallic", CUBE_METALLIC)
            tex_name = "procedural"
            material_meta = {
                "kind": "procedural", "material": "plastic",
                "color_name": "random", "primary_color": cube_rgb,
                "secondary_color": None, "pattern_type": "procedural_noise",
                "roughness": round(roughness, 4), "metallic": CUBE_METALLIC,
            }

        cube.replace_materials(mat)
        cube.set_cp("category_id", 1)

        idx = len(cubes)
        cube.set_cp("cube_index", idx)
        cube.set_cp("is_cube", True)
        cube.blender_obj.name = f"cube_{idx:03d}"
        positions.append((x, y, z))   # gerçek z (extra_z dahil)
        scales.append(scale)
        radii.append(effective_radius)
        cube_metas.append({
            "idx": idx, "scale": round(scale, 4),
            "pos_x": round(x, 4), "pos_y": round(y, 4), "pos_z": round(z, 4),
            "tilt_x": round(tilt_x, 4), "tilt_y": round(tilt_y, 4),
            "rotation_euler_rad": [round(value, 6) for value in rotation],
            "rotation_euler_deg": [round(math.degrees(value), 3) for value in rotation],
            "orientation_mode": orientation_mode,
            "half_extents": [round(float(value), 4) for value in half_extents],
            "texture": tex_name, "material": material_meta, "stacked_on": None,
        })
        cubes.append(cube)

    # Stacking: %STACK_PROBABILITY ihtimalle bir küp başkasının üstüne yerleşir
    if len(cubes) >= 2 and random.random() < STACK_PROBABILITY:
        stable = [
            index for index, meta in enumerate(cube_metas)
            if meta["orientation_mode"] != "strong_tilt"
        ]
        if len(stable) < 2:
            return cubes, cube_metas
        base_idx, stack_idx = random.sample(stable, 2)

        base_pos   = positions[base_idx]
        base_scale = scales[base_idx]
        stk_scale  = scales[stack_idx]
        base_rotation = tuple(cube_metas[base_idx]["rotation_euler_rad"])
        stack_rotation = tuple(cube_metas[stack_idx]["rotation_euler_rad"])
        base_half = _rotated_half_extents(base_scale, base_rotation)
        support_z = base_pos[2] + base_half[2]
        jitter_x = random.uniform(-STACK_XY_JITTER, STACK_XY_JITTER)
        jitter_y = random.uniform(-STACK_XY_JITTER, STACK_XY_JITTER)
        new_x = base_pos[0] + jitter_x
        new_y = base_pos[1] + jitter_y
        new_z, stack_half = _place_cube_on_support(
            cubes[stack_idx], new_x, new_y, stk_scale, stack_rotation, support_z
        )

        positions[stack_idx] = (new_x, new_y, new_z)
        cube_metas[stack_idx]["pos_x"] = round(new_x, 4)
        cube_metas[stack_idx]["pos_y"] = round(new_y, 4)
        cube_metas[stack_idx]["pos_z"] = round(new_z, 4)
        cube_metas[stack_idx]["half_extents"] = [
            round(float(value), 4) for value in stack_half
        ]
        cube_metas[stack_idx]["stacked_on"] = base_idx

    return cubes, cube_metas


# ── Etiketlenmeyen çevresel nesneler ────────────────────────────

def _environment_material(kind, color, metallic=None):
    mat = bproc.material.create(f"env_{kind}_{random.randint(0, 999_999)}")
    mat.set_principled_shader_value("Base Color", list(color) + [1.0])
    roughness = random.uniform(0.38, 0.88)
    metal = random.uniform(0.0, 0.35) if metallic is None else metallic
    mat.set_principled_shader_value("Roughness", roughness)
    mat.set_principled_shader_value("Metallic", metal)
    for key in ("Specular IOR Level", "Specular"):
        try:
            mat.set_principled_shader_value(key, random.uniform(0.18, 0.42))
            break
        except Exception:
            pass
    return mat, roughness, metal


def _tag_environment(obj, kind, index):
    """Çevre nesnesini açıkça arka plan sınıfına atar."""
    obj.set_cp("category_id", 0)
    obj.set_cp("is_cube", False)
    obj.set_cp("environment_type", kind)
    obj.blender_obj.pass_index = 0
    obj.blender_obj.name = f"env_{kind}_{index:03d}"


def _create_env_box(kind, location, scale, color, index, rotation_z=0.0):
    obj = bproc.object.create_primitive("CUBE")
    obj.set_scale(scale)
    obj.set_location(location)
    obj.set_rotation_euler([0.0, 0.0, rotation_z])
    mat, roughness, metallic = _environment_material(kind, color)
    obj.replace_materials(mat)
    _tag_environment(obj, kind, index)
    return obj, roughness, metallic


def _create_env_cylinder(kind, location, radius, height, color, index):
    obj = bproc.object.create_primitive("CYLINDER")
    obj.set_scale([radius, radius, height / 2.0])
    obj.set_location([location[0], location[1], location[2] + height / 2.0])
    mat, roughness, metallic = _environment_material(kind, color)
    obj.replace_materials(mat)
    _tag_environment(obj, kind, index)
    return obj, roughness, metallic


def _create_cable(start, end, radius, color, index):
    import mathutils  # type: ignore
    start_vec = mathutils.Vector(start)
    end_vec = mathutils.Vector(end)
    direction = end_vec - start_vec
    length = direction.length
    obj = bproc.object.create_primitive("CYLINDER")
    obj.set_scale([radius, radius, max(length / 2.0, radius)])
    obj.set_location(list((start_vec + end_vec) / 2.0))
    if length > FORWARD_VEC_NORM_THRESHOLD:
        obj.set_rotation_euler(direction.to_track_quat("Z", "Y").to_euler())
    mat, roughness, metallic = _environment_material("cable", color, metallic=0.0)
    obj.replace_materials(mat)
    _tag_environment(obj, "cable", index)
    return obj, roughness, metallic


def _sample_environment_xy(cube_metas, radius, near_cubes=False):
    area_x = FLOOR_WIDTH / 2.0 - radius
    area_y = FLOOR_HEIGHT / 2.0 - radius
    for _ in range(MAX_PLACE_TRIES):
        x = random.uniform(-max(area_x, 0.1), max(area_x, 0.1))
        y = random.uniform(-max(area_y, 0.1), max(area_y, 0.1))
        distances = []
        for cube in cube_metas:
            cube_radius = max(cube.get("half_extents", [cube["scale"]])[:2])
            distances.append(math.hypot(x - cube["pos_x"], y - cube["pos_y"]) - cube_radius)
        if not distances:
            return x, y
        min_distance = min(distances)
        if near_cubes:
            if radius + ENV_CUBE_CLEARANCE <= min_distance <= radius + 0.45:
                return x, y
        elif min_distance >= radius + ENV_CUBE_CLEARANCE:
            return x, y
    return None


def create_environment_objects(cube_metas, camera_location):
    """Etiketlenmeyen saha elemanları ve kontrollü kısmi örtücüler oluşturur."""
    if random.random() >= ENVIRONMENT_PROBABILITY:
        return [], []

    objects = []
    metadata = []
    type_probabilities = {
        "boundary_line": ENV_BOUNDARY_LINE_PROBABILITY,
        "barrier": ENV_BARRIER_PROBABILITY,
        "pole": ENV_POLE_PROBABILITY,
        "cable": ENV_CABLE_PROBABILITY,
        "cylinder": ENV_CYLINDER_PROBABILITY,
        "robot_part": ENV_ROBOT_PART_PROBABILITY,
        "shadow_caster": ENV_SHADOW_CASTER_PROBABILITY,
    }
    available = [kind for kind, probability in type_probabilities.items()
                 if random.random() < probability]
    if not available:
        return [], []
    count = random.randint(*ENV_OBJECT_COUNT_RANGE)

    for index in range(count):
        kind = random.choice(available)
        color = random.choice(ENV_COLORS)
        obj = None
        location = None
        dimensions = None
        roughness = metallic = None

        if kind == "boundary_line":
            width = random.uniform(*BOUNDARY_LINE_WIDTH_RANGE)
            horizontal = random.random() < 0.5
            if horizontal:
                y = random.choice([-1, 1]) * random.uniform(FLOOR_HEIGHT * 0.35, FLOOR_HEIGHT * 0.48)
                scale = [FLOOR_WIDTH * random.uniform(0.25, 0.48), width / 2, 0.004]
                location = [0.0, y, 0.006]
            else:
                x = random.choice([-1, 1]) * random.uniform(FLOOR_WIDTH * 0.35, FLOOR_WIDTH * 0.48)
                scale = [width / 2, FLOOR_HEIGHT * random.uniform(0.25, 0.48), 0.004]
                location = [x, 0.0, 0.006]
            obj, roughness, metallic = _create_env_box(
                kind, location, scale, color, index
            )
            dimensions = [round(value * 2, 4) for value in scale]

        elif kind == "barrier":
            length = random.uniform(*BARRIER_LENGTH_RANGE)
            width = random.uniform(0.04, 0.12)
            height = random.uniform(0.18, 0.65)
            edge_x = FLOOR_WIDTH / 2 - width
            edge_y = FLOOR_HEIGHT / 2 - width
            if random.random() < 0.5:
                location = [random.uniform(-edge_x, edge_x), random.choice([-edge_y, edge_y]), height / 2]
                scale = [length / 2, width / 2, height / 2]
            else:
                location = [random.choice([-edge_x, edge_x]), random.uniform(-edge_y, edge_y), height / 2]
                scale = [width / 2, length / 2, height / 2]
            obj, roughness, metallic = _create_env_box(kind, location, scale, color, index)
            dimensions = [round(value * 2, 4) for value in scale]

        elif kind in ("pole", "cylinder"):
            radius_range = POLE_RADIUS_RANGE if kind == "pole" else SMALL_CYLINDER_RADIUS_RANGE
            radius = random.uniform(*radius_range)
            height = (random.uniform(0.35, ENV_HEIGHT_RANGE[1]) if kind == "pole"
                      else random.uniform(*ENV_OBJECT_SIZE_RANGE))
            xy = _sample_environment_xy(cube_metas, radius, near_cubes=(kind == "pole"))
            if xy is not None:
                location = [xy[0], xy[1], 0.0]
                obj, roughness, metallic = _create_env_cylinder(
                    kind, location, radius, height, color, index
                )
                dimensions = [round(radius * 2, 4), round(radius * 2, 4), round(height, 4)]

        elif kind == "cable":
            radius = random.uniform(*CABLE_RADIUS_RANGE)
            start_xy = _sample_environment_xy(cube_metas, radius)
            end_xy = _sample_environment_xy(cube_metas, radius)
            if start_xy is not None and end_xy is not None:
                z = random.uniform(radius, radius * 4.0)
                start = [start_xy[0], start_xy[1], z]
                end = [end_xy[0], end_xy[1], z + random.uniform(-radius, radius)]
                obj, roughness, metallic = _create_cable(start, end, radius, color, index)
                location = [round((a + b) / 2, 4) for a, b in zip(start, end)]
                dimensions = [round(radius * 2, 4), round(math.dist(start, end), 4)]

        elif kind == "robot_part":
            size = random.uniform(*ENV_OBJECT_SIZE_RANGE)
            xy = _sample_environment_xy(cube_metas, size * 0.75)
            if xy is not None:
                scale = [size * random.uniform(0.7, 1.4), size * random.uniform(0.25, 0.55),
                         size * random.uniform(0.15, 0.40)]
                location = [xy[0], xy[1], scale[2]]
                obj, roughness, metallic = _create_env_box(
                    kind, location, scale, color, index,
                    rotation_z=random.uniform(0.0, 2.0 * math.pi),
                )
                dimensions = [round(value * 2, 4) for value in scale]

        elif kind == "shadow_caster":
            length = random.uniform(0.45, 1.25)
            width = random.uniform(0.08, 0.25)
            height = random.uniform(0.55, 1.4)
            xy = _sample_environment_xy(cube_metas, width)
            if xy is not None:
                scale = [length / 2, width / 2, 0.025]
                location = [xy[0], xy[1], height]
                obj, roughness, metallic = _create_env_box(
                    kind, location, scale, color, index,
                    rotation_z=random.uniform(0.0, 2.0 * math.pi),
                )
                dimensions = [round(value * 2, 4) for value in scale]

        if obj is not None:
            objects.append(obj)
            metadata.append({
                "type": kind,
                "location": [round(float(value), 4) for value in location],
                "dimensions": dimensions,
                "color": [round(value, 4) for value in color],
                "roughness": round(roughness, 4),
                "metallic": round(metallic, 4),
                "annotated": False,
            })

    # Kameraya göre hedef küpün önüne, temas etmeyen ince bir örtücü koy.
    if cube_metas and random.random() < ENV_CONTROLLED_OCCLUDER_PROBABILITY:
        target = random.choice(cube_metas)
        camera = np.asarray(camera_location, dtype=float)
        cube_pos = np.array([target["pos_x"], target["pos_y"], target["pos_z"]], dtype=float)
        direction = camera[:2] - cube_pos[:2]
        norm = np.linalg.norm(direction)
        if norm > FORWARD_VEC_NORM_THRESHOLD:
            direction /= norm
            radius = random.uniform(0.025, 0.060)
            cube_radius = max(target.get("half_extents", [target["scale"]])[:2])
            offset = cube_radius + radius + ENV_CUBE_CLEARANCE
            xy = cube_pos[:2] + direction * offset
            height = random.uniform(target["scale"] * 1.2, target["scale"] * 2.2)
            if abs(xy[0]) < FLOOR_WIDTH / 2 and abs(xy[1]) < FLOOR_HEIGHT / 2:
                index = len(objects)
                color = random.choice(ENV_COLORS)
                obj, roughness, metallic = _create_env_cylinder(
                    "controlled_occluder", [xy[0], xy[1], 0.0], radius, height,
                    color, index,
                )
                objects.append(obj)
                metadata.append({
                    "type": "controlled_occluder", "target_cube": target["idx"],
                    "location": [round(float(xy[0]), 4), round(float(xy[1]), 4), 0.0],
                    "dimensions": [round(radius * 2, 4), round(radius * 2, 4), round(height, 4)],
                    "color": [round(value, 4) for value in color],
                    "roughness": round(roughness, 4), "metallic": round(metallic, 4),
                    "annotated": False,
                })

    return objects, metadata


# ── Kamera fonksiyonları ─────────────────────────────────────────

def cubes_centroid(cubes):
    """Küplerin orta noktasını (centroid) hesaplar."""
    locs = np.array([c.get_location() for c in cubes], dtype=float)
    return locs.mean(axis=0)


def place_camera(cubes):
    """
    Küplerin centroid'ine bakan, rastgele konumlandırılmış kamera ekler.
    Aim jitter ve Depth of Field destekler.

    Returns:
        dict: {distance, elevation_deg, azimuth_deg, focal_mm, dof_enabled, dof_fstop}
    """
    # Hedef noktayı centroid + küçük rastgele sapma
    jx = random.uniform(-CAMERA_AIM_JITTER, CAMERA_AIM_JITTER)
    jy = random.uniform(-CAMERA_AIM_JITTER, CAMERA_AIM_JITTER)
    target  = cubes_centroid(cubes) + np.array([jx, jy, 0.0])

    # Yükseklik, uzaklık ve elevation sınırlarının hepsini karşılayan örnek.
    for _ in range(100):
        dist = random.uniform(MIN_DISTANCE, MAX_DISTANCE)
        height = random.uniform(CAMERA_HEIGHT_MIN, min(CAMERA_HEIGHT_MAX, dist * 0.995))
        elevation_rad = math.asin(min(0.999, height / dist))
        elevation_deg = math.degrees(elevation_rad)
        if MIN_ELEVATION <= elevation_deg <= MAX_ELEVATION:
            break
    else:
        dist = random.uniform(MIN_DISTANCE, MAX_DISTANCE)
        elevation_rad = math.radians(random.uniform(MIN_ELEVATION, MAX_ELEVATION))
        height = dist * math.sin(elevation_rad)

    azimuth_rad = math.radians(random.uniform(0.0, 360.0))
    horizontal = math.sqrt(max(dist * dist - height * height, 0.01))
    cam_offset = np.array([
        horizontal * math.cos(azimuth_rad),
        horizontal * math.sin(azimuth_rad),
        height,
    ])
    cam_loc = target + cam_offset

    forward = target - cam_loc
    norm    = np.linalg.norm(forward)
    forward = forward / norm if norm > FORWARD_VEC_NORM_THRESHOLD else np.array([0.0, 0.0, -1.0])

    roll = random.uniform(CAMERA_INPLANE_ROT_MIN, CAMERA_INPLANE_ROT_MAX)
    cam_rot = bproc.camera.rotation_from_forward_vec(forward, inplane_rot=roll)

    focal_mm = random.uniform(FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=focal_mm,
        image_width=IMG_RESOLUTION,
        image_height=IMG_RESOLUTION,
        lens_unit="MILLIMETERS",
    )

    cam_matrix = bproc.math.build_transformation_mat(cam_loc, cam_rot)
    bproc.camera.add_camera_pose(cam_matrix)

    # Derinlik alanı (DoF / Bokeh)
    dof_on    = False
    dof_fstop = None
    if USE_DOF and random.random() < DOF_PROBABILITY:
        try:
            import bpy  # type: ignore
            cam_data = bpy.context.scene.camera.data
            cam_data.dof.use_dof = True
            dof_fstop = random.uniform(DOF_FSTOP_MIN, DOF_FSTOP_MAX)
            cam_data.dof.aperture_fstop = dof_fstop
            cam_data.dof.focus_distance = float(np.linalg.norm(target - cam_loc))
            dof_on = True
        except Exception:
            pass
    else:
        try:
            import bpy  # type: ignore
            bpy.context.scene.camera.data.dof.use_dof = False
        except Exception:
            pass

    return {
        "distance":      round(dist, 3),
        "height":        round(float(cam_loc[2]), 3),
        "elevation_deg": round(math.degrees(elevation_rad), 2),
        "azimuth_deg":   round(math.degrees(azimuth_rad), 2),
        "roll_deg":      round(math.degrees(roll), 2),
        "focal_mm":      round(focal_mm, 1),
        "location":      [round(float(value), 4) for value in cam_loc],
        "target":        [round(float(value), 4) for value in target],
        "aim_jitter":    [round(jx, 3), round(jy, 3)],
        "dof_enabled":   dof_on,
        "dof_fstop":     round(dof_fstop, 2) if dof_fstop else None,
    }


# ── Bbox & görünürlük yardımcı fonksiyonları ────────────────────

def _eevee_bbox_2d(cube):
    """
    Küpün 3D bounding box'ını kamera matrisine göre 2D piksel koordinatlarına
    projekte eder. Frustum dışındaysa None döner.

    Returns:
        tuple(x, y, w, h) piksel cinsinden, ya da None
    """
    import bpy        # type: ignore
    import mathutils  # type: ignore

    cam  = bpy.context.scene.camera
    dg   = bpy.context.evaluated_depsgraph_get()
    mv   = cam.matrix_world.inverted()
    proj = cam.calc_matrix_camera(dg, x=IMG_RESOLUTION, y=IMG_RESOLUTION)

    pts = []
    for corner in cube.blender_obj.bound_box:
        world = cube.blender_obj.matrix_world @ mathutils.Vector(corner)
        ndc   = proj @ mv @ world.to_4d()
        if ndc.w <= 0:
            return None
        x = ( ndc.x / ndc.w + 1.0) / 2.0 * IMG_RESOLUTION
        y = (1.0 - (ndc.y / ndc.w + 1.0) / 2.0) * IMG_RESOLUTION
        pts.append((x, y))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = max(0.0, min(xs))
    y0 = max(0.0, min(ys))
    x1 = min(float(IMG_RESOLUTION), max(xs))
    y1 = min(float(IMG_RESOLUTION), max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _compute_eevee_visibility(bboxes_px, cam_distances):
    """
    Her küpün tahmini görünürlük oranını 2D bbox örtüşmesine göre hesaplar.
    Kameraya daha yakın küpler, gerideki küplerin görünürlüğünü azaltır.

    Args:
        bboxes_px:     [(x, y, w, h), ...] — piksel cinsinden (None = frustum dışı)
        cam_distances: [float, ...]        — her küpün kameraya uzaklığı

    Returns:
        list[float]: Her küp için [0, 1] görünürlük oranı
    """
    n          = len(bboxes_px)
    visibility = [1.0] * n
    # Uzaktan yakına sırala: arkadaki küpler önce işlenir
    order = sorted(range(n), key=lambda i: -cam_distances[i])

    for idx, i in enumerate(order):
        if bboxes_px[i] is None:
            visibility[i] = 0.0
            continue
        x0, y0, w0, h0 = bboxes_px[i]
        area_i = w0 * h0
        if area_i <= 0:
            visibility[i] = 0.0
            continue

        occluded = 0.0
        for j in order[:idx]:  # Kameraya daha yakın tüm küpler
            if bboxes_px[j] is None:
                continue
            x1, y1, w1, h1 = bboxes_px[j]
            ix = max(0.0, min(x0 + w0, x1 + w1) - max(x0, x1))
            iy = max(0.0, min(y0 + h0, y1 + h1) - max(y0, y1))
            occluded += ix * iy

        visibility[i] = max(0.0, 1.0 - occluded / area_i)

    return visibility


def _is_valid_annotation(w_n, h_n, visibility_ratio):
    """
    Bir bbox'ın anotasyon için geçerli olup olmadığını kontrol eder.

    - w_n, h_n:          Normalize bbox boyutları [0, 1]
    - visibility_ratio:  Küpün görünen oranı [0, 1]
    """
    area_px = (w_n * IMG_RESOLUTION) * (h_n * IMG_RESOLUTION)
    return (
        area_px >= MIN_BBOX_AREA_PX
        and visibility_ratio >= MIN_VISIBILITY_RATIO
        and 0.001 < w_n < 1.0
        and 0.001 < h_n < 1.0
    )


_worker_np_rng = np.random.default_rng(RANDOM_SEED)


def apply_post_augmentation(img_arr, auxiliary_arrays=None):
    """
    Render sonrası PIL tabanlı görüntü augmentasyonu.
    Gerçek kamera sensör efektlerini simüle eder.

    Args:
        img_arr: uint8 HxWx3 numpy dizisi

    Returns:
        tuple: (augmented_uint8_array, aug_meta_dict)
    """
    from image_effects import process_image

    config = {
        "enabled": POST_AUGMENTATION,
        "noise_probability": POST_AUG_NOISE_PROB,
        "noise_sigma_range": (POST_AUG_NOISE_MIN, POST_AUG_NOISE_MAX),
        "gaussian_blur_probability": POST_AUG_BLUR_PROB,
        "gaussian_blur_range": (POST_AUG_BLUR_MIN, POST_AUG_BLUR_MAX),
        "temperature_probability": POST_AUG_WB_PROB,
        "temperature_range": POST_AUG_TEMPERATURE_RANGE,
        "brightness_contrast_probability": POST_AUG_BC_PROB,
        "brightness_range": (POST_AUG_BRIGHTNESS_MIN, POST_AUG_BRIGHTNESS_MAX),
        "contrast_range": (POST_AUG_CONTRAST_MIN, POST_AUG_CONTRAST_MAX),
        "gamma_probability": POST_AUG_GAMMA_PROB,
        "gamma_range": POST_AUG_GAMMA_RANGE,
        "motion_blur_probability": POST_AUG_MOTION_BLUR_PROB,
        "motion_blur_radius_range": POST_AUG_MOTION_BLUR_RADIUS_RANGE,
        "lens_distortion_probability": POST_AUG_LENS_DISTORTION_PROB,
        "lens_distortion_range": POST_AUG_LENS_DISTORTION_RANGE,
        "exposure_probability": POST_AUG_EXPOSURE_PROB,
        "exposure_ev_range": POST_AUG_EXPOSURE_EV_RANGE,
        "low_res_probability": LOW_RES_PROBABILITY,
        "low_res_sizes": LOW_RES_SIZES,
        "upscale_methods": LOW_RES_UPSCALE_METHODS,
        "jpeg_probability": JPEG_COMPRESSION_PROBABILITY,
        "jpeg_quality_range": JPEG_QUALITY_RANGE,
    }
    return process_image(
        img_arr, config, rng=random, np_rng=_worker_np_rng,
        auxiliary_arrays=auxiliary_arrays,
    )


# ── Frame kaydetme fonksiyonları ─────────────────────────────────

def _save_eevee_frame(colors, cubes, frame_idx, yolo_img_dir, yolo_lbl_dir):
    """
    Eevee render sonucunu YOLO formatında kaydeder.
    3D→2D projeksiyon + 2D örtüşme tabanlı görünürlük filtresi kullanır.

    Returns:
        tuple: (n_annotated, n_skipped, vis_ratios, aug_meta)
    """
    from PIL import Image as PILImage
    import bpy  # type: ignore

    img_arr = colors[0]
    if img_arr.dtype != np.uint8:
        img_arr = (np.clip(img_arr, 0.0, 1.0) * 255).astype(np.uint8)
    if img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3]

    # Post-processing augmentation
    img_arr, aug_meta, _ = apply_post_augmentation(img_arr)

    stem     = f"{frame_idx:06d}"
    img_path = os.path.join(yolo_img_dir, f"{stem}.jpg")
    lbl_path = os.path.join(yolo_lbl_dir, f"{stem}.txt")

    PILImage.fromarray(img_arr).save(img_path, "JPEG", quality=95)

    # Her küpün bbox'ını ve kameraya uzaklığını hesapla
    cam_loc   = np.array(bpy.context.scene.camera.location)
    bboxes_px = [_eevee_bbox_2d(cube) for cube in cubes]
    lens_k = float(aug_meta.get("lens_distortion_k", 0.0))
    if abs(lens_k) > 0:
        from image_effects import distort_bbox
        bboxes_px = [
            distort_bbox(bbox, lens_k, IMG_RESOLUTION, IMG_RESOLUTION)
            for bbox in bboxes_px
        ]
    cam_dists = [
        float(np.linalg.norm(
            np.array(cube.blender_obj.matrix_world.translation) - cam_loc
        ))
        for cube in cubes
    ]

    vis_ratios = _compute_eevee_visibility(bboxes_px, cam_dists)

    lines       = []
    n_annotated = 0
    n_skipped   = 0
    annotation_records = []

    for i, cube in enumerate(cubes):
        bbox = bboxes_px[i]
        if bbox is None:
            vis_ratios[i] = 0.0
            n_skipped += 1
            continue
        x, y, w, h = bbox
        cx = max(0.0, min(1.0, (x + w / 2) / IMG_RESOLUTION))
        cy = max(0.0, min(1.0, (y + h / 2) / IMG_RESOLUTION))
        nw = max(0.0, min(1.0, w / IMG_RESOLUTION))
        nh = max(0.0, min(1.0, h / IMG_RESOLUTION))

        if not _is_valid_annotation(nw, nh, vis_ratios[i]):
            n_skipped += 1
            continue

        lines.append(
            f"0 {cx:.{YOLO_DECIMAL_PLACES}f} {cy:.{YOLO_DECIMAL_PLACES}f}"
            f" {nw:.{YOLO_DECIMAL_PLACES}f} {nh:.{YOLO_DECIMAL_PLACES}f}"
        )
        annotation_records.append({
            "cube_idx": i,
            "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
            "area": round(w * h, 3),
            "visibility_ratio": round(vis_ratios[i], 4),
        })
        n_annotated += 1

    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return n_annotated, n_skipped, vis_ratios, aug_meta, annotation_records


def _cube_instance_mapping(attribute_map, cubes):
    """BlenderProc sürümleri arasında instance id → küp index eşlemesi."""
    mapping = {}
    records = attribute_map or []
    if isinstance(records, dict):
        records = list(records.values())
    for record in records:
        if not isinstance(record, dict):
            continue
        instance_id = record.get("idx", record.get("instance_id", record.get("id")))
        if instance_id is None:
            continue
        name = str(record.get("name", ""))
        cube_index = record.get("cube_index", record.get("cp_cube_index"))
        if cube_index is None and name.startswith("cube_"):
            try:
                cube_index = int(name.rsplit("_", 1)[1])
            except ValueError:
                cube_index = None
        if cube_index is not None and 0 <= int(cube_index) < len(cubes):
            mapping[int(instance_id)] = int(cube_index)

    # Eski BlenderProc çıktısında attribute map yoksa pass_index düzenini koru.
    if not mapping:
        mapping = {index + 1: index for index in range(len(cubes))}
    return mapping


def _save_cycles_frame(colors, instance_segmaps, cubes, frame_idx, yolo_img_dir,
                       yolo_lbl_dir, instance_attribute_map=None):
    """
    Cycles render sonucunu YOLO formatında kaydeder.
    Instance segmap tabanlı kesin görünürlük hesabı yapar.
    %40'tan az görünen küpler anotate edilmez.

    Returns:
        tuple: (n_annotated, n_skipped, vis_ratios, aug_meta)
    """
    from PIL import Image as PILImage

    img_arr = colors[0]
    if img_arr.dtype != np.uint8:
        img_arr = (np.clip(img_arr, 0.0, 1.0) * 255).astype(np.uint8)
    if img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3]

    raw_segmap = instance_segmaps[0].astype(np.int32)
    img_arr, aug_meta, aux = apply_post_augmentation(
        img_arr, auxiliary_arrays=[raw_segmap]
    )

    stem = f"{frame_idx:06d}"
    PILImage.fromarray(img_arr).save(
        os.path.join(yolo_img_dir, f"{stem}.jpg"), "JPEG", quality=95
    )

    segmap = aux[0].astype(np.int32) if aux else raw_segmap
    h_img, w_img = segmap.shape
    lines       = []
    n_annotated = 0
    n_skipped   = 0
    vis_ratios  = [-1.0] * len(cubes)  # -1 = inst_id bulunamadı
    annotation_records = []
    instance_to_cube = _cube_instance_mapping(instance_attribute_map, cubes)
    cube_to_instance = {cube_idx: inst_id for inst_id, cube_idx in instance_to_cube.items()}

    for cube_idx in range(len(cubes)):
        inst_id = cube_to_instance.get(cube_idx)
        if inst_id is None or not np.any(segmap == inst_id):
            vis_ratios[cube_idx] = 0.0
            n_skipped += 1
            continue
        mask = segmap == inst_id
        visible_pixels = int(np.sum(mask))
        full_bbox = _eevee_bbox_2d(cubes[cube_idx])
        if full_bbox is None:
            n_skipped += 1
            vis_ratios[cube_idx] = 0.0
            continue
        full_area = full_bbox[2] * full_bbox[3]
        vis_ratio = min(1.0, visible_pixels / max(full_area, 1.0))
        vis_ratios[cube_idx] = round(vis_ratio, 4)

        rows  = np.where(mask.any(axis=1))[0]
        cols  = np.where(mask.any(axis=0))[0]
        y_min, y_max = int(rows[0]), int(rows[-1])
        x_min, x_max = int(cols[0]), int(cols[-1])
        cx = max(0.0, min(1.0, (x_min + x_max + 1) / 2 / w_img))
        cy = max(0.0, min(1.0, (y_min + y_max + 1) / 2 / h_img))
        nw = max(0.0, min(1.0, (x_max - x_min + 1) / w_img))
        nh = max(0.0, min(1.0, (y_max - y_min + 1) / h_img))

        if not _is_valid_annotation(nw, nh, vis_ratio):
            n_skipped += 1
            continue

        lines.append(
            f"0 {cx:.{YOLO_DECIMAL_PLACES}f} {cy:.{YOLO_DECIMAL_PLACES}f}"
            f" {nw:.{YOLO_DECIMAL_PLACES}f} {nh:.{YOLO_DECIMAL_PLACES}f}"
        )
        annotation_records.append({
            "cube_idx": cube_idx,
            "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
            "area": visible_pixels,
            "visibility_ratio": round(vis_ratio, 4),
        })
        n_annotated += 1

    with open(os.path.join(yolo_lbl_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return n_annotated, n_skipped, vis_ratios, aug_meta, annotation_records


# ── Dataset dosyaları ────────────────────────────────────────────

def create_coco_document():
    return {
        "info": {
            "description": "BlenderProc synthetic cube dataset",
            "version": "2.0",
            "date_created": datetime.now().isoformat(timespec="seconds"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "cube", "supercategory": "object"}],
    }


def append_coco_frame(document, frame_idx, annotation_records, image_file_name):
    image_id = frame_idx + 1
    document["images"].append({
        "id": image_id, "file_name": image_file_name,
        "width": IMG_RESOLUTION, "height": IMG_RESOLUTION,
    })
    next_annotation_id = len(document["annotations"]) + 1
    for record in annotation_records:
        x, y, width, height = [float(value) for value in record["bbox"]]
        document["annotations"].append({
            "id": next_annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [round(x, 3), round(y, 3), round(width, 3), round(height, 3)],
            "area": round(float(record.get("area", width * height)), 3),
            "segmentation": [],
            "iscrowd": 0,
            "cube_index": int(record["cube_idx"]),
            "visibility_ratio": float(record.get("visibility_ratio", 1.0)),
        })
        next_annotation_id += 1


def write_coco_annotations(document, coco_dir):
    """COCO JSON'ı yarım yazılmış dosya bırakmadan atomik olarak günceller."""
    path = os.path.join(coco_dir, COCO_ANNOTATIONS_FILE)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def write_dataset_yaml(yolo_dir, total_images):
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    abs_yolo  = os.path.abspath(yolo_dir)

    content = (
        f"# Otomatik üretildi — generate.py\n"
        f"path: {abs_yolo}\n"
        f"train: images  # Toplam {total_images} görüntünün ~{int(DATASET_TRAIN_RATIO*100)}'i için train split önerilir\n"
        f"val:   images  # Toplam {total_images} görüntünün ~{int(DATASET_VAL_RATIO*100)}'si için val split önerilir\n"
        f"\n"
        f"nc: {DATASET_NC}\n"
        f"names: {DATASET_NAMES}\n"
    )

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"dataset.yaml yazıldı → {yaml_path}")


# ── Ana üretim döngüsü ───────────────────────────────────────────

def main():
    """Sentetik veri üretim ana döngüsü."""
    global _worker_np_rng
    if _num_workers < 1:
        raise ValueError("--num-workers en az 1 olmalıdır.")
    if not 0 <= _worker_id < _num_workers:
        raise ValueError("--worker-id, 0 ile --num-workers-1 arasında olmalıdır.")
    worker_seed = RANDOM_SEED + _worker_id * 1_000_003
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))
    _worker_np_rng = np.random.default_rng(worker_seed)

    if _num_workers > 1:
        base = NUM_IMAGES // _num_workers
        _images_to_generate = base + (NUM_IMAGES % _num_workers if _worker_id == _num_workers - 1 else 0)
        worker_root = os.path.join(OUTPUT_DIR, "workers", f"worker_{_worker_id}")
        worker_yolo = os.path.join(worker_root, YOLO_SUBDIR)
        print(f"Worker {_worker_id}/{_num_workers}: {_images_to_generate} görüntü üretilecek")
    else:
        worker_root = OUTPUT_DIR
        _images_to_generate = NUM_IMAGES
        worker_yolo = os.path.join(OUTPUT_DIR, YOLO_SUBDIR)

    worker_coco = os.path.join(worker_root, COCO_SUBDIR)
    coco_img_dir = os.path.join(worker_coco, COCO_IMAGES_SUBDIR)

    yolo_img_dir  = os.path.join(worker_yolo, YOLO_IMAGES_SUBDIR)
    yolo_lbl_dir  = os.path.join(worker_yolo, YOLO_LABELS_SUBDIR)
    yolo_meta_dir = os.path.join(worker_yolo, YOLO_META_SUBDIR)

    setup_output_dirs(yolo_img_dir, yolo_lbl_dir, yolo_meta_dir)
    os.makedirs(coco_img_dir, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR), exist_ok=True)  # errors.log için

    hdri_files = sorted({
        path
        for directory in BACKGROUND_ASSET_DIRS
        for path in scan_files(directory, HDRI_EXTENSIONS)
    })
    image_background_files = sorted({
        path
        for directory in BACKGROUND_ASSET_DIRS
        for path in scan_files(directory, IMAGE_BACKGROUND_EXTENSIONS)
    })

    print(f"HDRI dosyası bulundu    : {len(hdri_files)}")
    print(f"JPG/PNG arka planı bulundu: {len(image_background_files)}")
    print(f"Toplam ortak arka plan havuzu: {len(hdri_files) + len(image_background_files)}")
    print(
        "Taranan arka plan klasörleri: "
        + ", ".join(os.path.relpath(path, BASE_DIR) for path in BACKGROUND_ASSET_DIRS)
    )
    if hdri_files and not image_background_files:
        print(
            "UYARI: JPG/JPEG/PNG bulunamadı; bu çalışmada bütün kareler HDRI olacak. "
            "Görselleri backgrounds/ veya textures/ içine koyun."
        )
    elif image_background_files and not hdri_files:
        print(
            "UYARI: EXR/HDR bulunamadı; bu çalışmada bütün kareler JPG/PNG olacak."
        )
    print("Arka plan seçimi        : Ortak havuzdaki her dosya eşit şanslı")
    print(f"Hedef görüntü sayısı    : {_images_to_generate}")
    print(f"Render samples          : {RENDER_SAMPLES}")
    print(f"Kalite profili          : {QUALITY_PROFILE}")
    print(f"Worker seed             : {worker_seed}")
    print(f"Düşük çözünürlük olasılığı: {LOW_RES_PROBABILITY:.2f}")

    if _regen_textures and os.path.isdir(TEXTURE_POOL_DIR):
        shutil.rmtree(TEXTURE_POOL_DIR)
        print(f"Eski texture havuzu silindi: {TEXTURE_POOL_DIR}")

    fdm_pool = generate_texture_pool() if USE_FDM_TEXTURES else None
    if fdm_pool:
        print(f"FDM texture havuzu      : {len(fdm_pool)} doku")

    if _only_textures:
        print("Texture üretimi tamamlandı. Render yapılmadı (--gen-textures).")
        return

    bproc.init()

    import bpy  # type: ignore

    render_settings = {}
    selected_backend = None
    if _use_eevee:
        for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                bpy.context.scene.render.engine = engine
                break
            except TypeError:
                continue
        eevee = getattr(bpy.context.scene, "eevee", None)
        if eevee is not None:
            _safe_setattr(eevee, "taa_render_samples", RENDER_SAMPLES)
            _safe_setattr(eevee, "use_gtao", True)
            _safe_setattr(eevee, "gtao_distance", 0.5)
            _safe_setattr(eevee, "gtao_factor", 1.35)
            _safe_setattr(eevee, "use_bloom", False)
        render_settings = {
            "profile": QUALITY_PROFILE, "samples": RENDER_SAMPLES,
            "color_management": _configure_color_management(bpy.context.scene),
        }
        print(f"Eevee render motoru etkinleştirildi ({engine}) — GPU otomatik.")
    else:
        if _use_optix or _use_gpu or _use_hybrid:
            try:
                selected_backend = _configure_cycles_devices(bpy)
            except Exception as gpu_err:
                print(f"GPU etkinleştirilemedi, CPU kullanılıyor: {gpu_err}")
        render_settings = _configure_cycles_quality(bpy, selected_backend)
        render_settings["device_backend"] = selected_backend or "auto/cpu"

    if not _use_eevee:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_segmentation_output(
            map_by=["instance", "class", "name"],
            default_values={"category_id": 0},
        )

    bproc.camera.set_resolution(IMG_RESOLUTION, IMG_RESOLUTION)

    successful = 0
    frame_idx  = 0
    coco_document = create_coco_document()

    while successful < _images_to_generate:
        frame_idx  += 1
        cubes       = []
        cube_metas  = []
        lights      = []
        light_meta  = []
        environment_objects = []
        environment_meta = []
        background_object = None
        floor       = None
        floor_color = None
        floor_type  = "solid"

        if successful % RENDER_PROGRESS_INTERVAL == 0:
            print(f"[{successful + 1}/{_images_to_generate}] Frame {frame_idx} render ediliyor...", flush=True)

        try:
            bproc.utility.reset_keyframes()

            frame_seed = worker_seed + frame_idx * 1009
            random.seed(frame_seed)
            _worker_np_rng = np.random.default_rng(frame_seed)
            background_source = select_background_source(
                hdri_files, image_background_files
            )
            background_meta = setup_background(background_source)
            lights, light_meta = create_scene_lights()
            floor, floor_color, floor_meta = create_floor(background_source)
            floor_type = floor_meta["type"]
            if floor is not None:
                floor.blender_obj.pass_index = 0
                floor.set_cp("category_id", 0)
                floor.set_cp("is_cube", False)

            cubes, cube_metas = create_cubes(floor_color, fdm_pool=fdm_pool)
            if not cubes:
                raise ValueError("Hiçbir küp yerleştirilemedi.")
            for i, cube in enumerate(cubes):
                cube.blender_obj.pass_index = i + 1

            cam_meta = place_camera(cubes)
            background_object, image_background_meta = create_image_backdrop(
                background_source
            )
            background_meta.update(image_background_meta)
            environment_objects, environment_meta = create_environment_objects(
                cube_metas, cam_meta["location"]
            )
            data     = bproc.renderer.render()

            if _use_eevee:
                n_ann, n_skip, vis_ratios, aug_meta, annotation_records = _save_eevee_frame(
                    data["colors"], cubes, successful, yolo_img_dir, yolo_lbl_dir
                )
            else:
                attribute_maps = data.get("instance_attribute_maps", [])
                attribute_map = attribute_maps[0] if attribute_maps else None
                n_ann, n_skip, vis_ratios, aug_meta, annotation_records = _save_cycles_frame(
                    data["colors"], data["instance_segmaps"], cubes,
                    successful, yolo_img_dir, yolo_lbl_dir, attribute_map
                )

            stem = f"{successful:06d}"
            image_name = f"{stem}.jpg"
            shutil.copy2(
                os.path.join(yolo_img_dir, image_name),
                os.path.join(coco_img_dir, image_name),
            )
            append_coco_frame(
                coco_document, successful, annotation_records,
                f"{COCO_IMAGES_SUBDIR}/{image_name}",
            )
            write_coco_annotations(coco_document, worker_coco)
            annotated_cube_indices = {record["cube_idx"] for record in annotation_records}

            # Per-frame metadata kaydet
            frame_meta = {
                "frame":   successful,
                "engine":  "eevee" if _use_eevee else "cycles",
                "hdri": background_meta["name"]
                    if background_meta["source_type"] == "hdri" else None,
                "background": background_meta,
                "background_source_type": background_meta["source_type"],
                "background_file": background_meta["name"],
                "camera":  cam_meta,
                "floor_type": floor_type,
                "floor": floor_meta,
                "lights": light_meta,
                "environment_objects": environment_meta,
                "n_environment_objects": len(environment_meta),
                "quality": render_settings,
                "seed": frame_seed,
                "n_cubes_total":            len(cubes),
                "n_cubes_annotated":        n_ann,
                "n_cubes_skipped":          n_skip,
                "post_aug":                 aug_meta,
                "cubes": [
                    {
                        **cm,
                        "visibility_ratio": round(vis_ratios[cm["idx"]], 4)
                            if cm["idx"] < len(vis_ratios) and vis_ratios[cm["idx"]] >= 0 else None,
                        "annotated": cm["idx"] in annotated_cube_indices,
                    }
                    for cm in cube_metas
                ],
            }
            meta_path = os.path.join(yolo_meta_dir, f"{successful:06d}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(frame_meta, f, indent=2, ensure_ascii=False)

            successful += 1
            print(
                f"  Tamamlandı [{successful}/{_images_to_generate}] "
                f"Küp: {len(cubes)} ann:{n_ann} skip:{n_skip} | "
                f"Env:{len(environment_meta)} | Zemin: {floor_type} | "
                f"Arka plan: {background_meta['source_type']}/"
                f"{background_meta['name']}",
                flush=True,
            )

        except Exception:
            log_error(frame_idx, traceback.format_exc())

        finally:
            if cubes:
                try:
                    bproc.object.delete_multiple(cubes)
                except Exception:
                    pass
            if floor is not None:
                try:
                    floor.delete()
                except Exception:
                    pass
            if environment_objects:
                try:
                    bproc.object.delete_multiple(environment_objects)
                except Exception:
                    for obj in environment_objects:
                        try:
                            obj.delete()
                        except Exception:
                            pass
            if background_object is not None:
                try:
                    background_object.delete()
                except Exception:
                    pass
            for light in lights:
                try:
                    light.delete()
                except Exception:
                    pass

    print(f"\nToplam başarılı render: {successful}")
    write_coco_annotations(coco_document, worker_coco)

    if _num_workers > 1:
        print(f"Worker {_worker_id}/{_num_workers} tamamlandı → {yolo_lbl_dir}")
        print("Tüm workerlar bitince merge için: python merge.py")
    else:
        write_dataset_yaml(os.path.join(OUTPUT_DIR, YOLO_SUBDIR), successful)


if __name__ == "__main__":
    main()
# ================================================================
# GÜNCEL PROJEYİ ÇALIŞTIRMA REHBERİ
# ================================================================
#
# ARKA PLAN DOSYALARININ YERİ
# ------------------------------------------------
# EXR/HDR ve JPG/JPEG/PNG dosyaları aşağıdaki klasörlere konulabilir:
#
#   backgrounds/
#   textures/
#
# İki klasör de alt klasörleriyle birlikte taranır.
# Her görüntüde yalnızca TEK arka plan kullanılır:
#   - ya bir EXR/HDR
#   - ya bir JPG/JPEG/PNG
#
# Program başlarken bulunan dosya sayılarını kontrol edin:
#
#   HDRI dosyası bulundu      : N
#   JPG/PNG arka planı bulundu: N
#
# İstenen karışık arka plan üretimi için iki sayı da 0'dan büyük olmalıdır.
#
#
# 1) FDM TEXTURE HAVUZU
# ------------------------------------------------
# Bu adım zorunlu değildir. Havuz yoksa generate.py otomatik oluşturur.
# Önceden oluşturmak render başlangıcını hızlandırır.
#
# Varsayılan 200 adet, 512x512 texture:
#
#   python gen_textures.py
#
# Belirli ayarlarla oluştur:
#
#   python gen_textures.py --pool-size 200 --size 512 --seed 7
#
# Eski havuzu silip yeniden oluştur:
#
#   python gen_textures.py --pool-size 200 --size 512 --seed 7 --regen
#
# BlenderProc üzerinden texture havuzunu yeniden oluşturup render başlat:
#
#   blenderproc run generate.py -- --regen-textures
#
# Yalnızca texture oluştur, render yapma:
#
#   blenderproc run generate.py -- --gen-textures
#
#
# 2) HIZLI TEST — 10 GÖRÜNTÜ
# ------------------------------------------------
# --test seçeneği:
#   - 10 görüntü üretir
#   - sample sayısını en fazla 16 yapar
#   - texture havuz boyutunu 10 yapar
#
# Windows PowerShell ve Linux:
#
#   blenderproc run generate.py -- --test --quality-profile fast --seed 42
#
# Render sonrasında bbox görsellerini ve dataset'i kontrol et:
#
#   python draw_bbox.py --limit 10
#   python validate_dataset.py
#
# Bbox görselleri:
#
#   output/bbox_debug/
#
#
# 3) TEK WORKER — CYCLES + OPTIX
# ------------------------------------------------
# NVIDIA RTX/OptiX için önerilen dengeli üretim:
#
#   blenderproc run generate.py -- --optix \
#     --quality-profile balanced \
#     --num-images 1000 \
#     --seed 42
#
# Windows PowerShell tek satır:
#
#   blenderproc run generate.py -- --optix --quality-profile balanced --num-images 1000 --seed 42
#
#
# 4) TEK WORKER — CYCLES + CUDA
# ------------------------------------------------
# NVIDIA GPU ile CUDA:
#
#   blenderproc run generate.py -- --gpu \
#     --quality-profile balanced \
#     --num-images 1000 \
#     --seed 42
#
# Windows PowerShell tek satır:
#
#   blenderproc run generate.py -- --gpu --quality-profile balanced --num-images 1000 --seed 42
#
#
# 5) CPU + GPU HİBRİT
# ------------------------------------------------
# CUDA üzerinden CPU ve GPU birlikte:
#
#   blenderproc run generate.py -- --hybrid \
#     --quality-profile balanced \
#     --num-images 1000 \
#     --seed 42
#
# Windows PowerShell tek satır:
#
#   blenderproc run generate.py -- --hybrid --quality-profile balanced --num-images 1000 --seed 42
#
#
# 6) EEVEE — HIZLI ÜRETİM
# ------------------------------------------------
# Eevee hızlıdır ancak görünürlük/örtülme hesabı Cycles kadar kesin değildir.
#
#   blenderproc run generate.py -- --eevee \
#     --quality-profile fast \
#     --num-images 1000 \
#     --seed 42
#
# Windows PowerShell tek satır:
#
#   blenderproc run generate.py -- --eevee --quality-profile fast --num-images 1000 --seed 42
#
#
# 7) CYCLES — GERÇEKÇİ KALİTE
# ------------------------------------------------
# Daha kaliteli fakat daha yavaş üretim:
#
#   blenderproc run generate.py -- --optix \
#     --quality-profile realistic \
#     --num-images 1000 \
#     --seed 42
#
# Windows PowerShell tek satır:
#
#   blenderproc run generate.py -- --optix --quality-profile realistic --num-images 1000 --seed 42
#
#
# 8) DÜŞÜK ÇÖZÜNÜRLÜK VE JPEG AYARLARIYLA ÜRETİM
# ------------------------------------------------
#
# Linux:
#
#   blenderproc run generate.py -- --optix \
#     --quality-profile balanced \
#     --num-images 1000 \
#     --seed 42 \
#     --low-res-probability 0.35 \
#     --low-res-sizes 160,240,320,480 \
#     --jpeg-compression-probability 0.25 \
#     --jpeg-quality-range 58,90
#
# Windows PowerShell tek satır:
#
#   blenderproc run generate.py -- --optix --quality-profile balanced --num-images 1000 --seed 42 --low-res-probability 0.35 --low-res-sizes 160,240,320,480 --jpeg-compression-probability 0.25 --jpeg-quality-range 58,90
#
#
# 9) LINUX — 4 PARALEL WORKER
# ------------------------------------------------
# --num-images 1000 toplam görüntü sayısıdır.
# Dört worker bu sayıyı aralarında paylaşır: yaklaşık 250'şer görüntü.
#
#   for i in 0 1 2 3; do
#     blenderproc run generate.py -- \
#       --optix \
#       --quality-profile balanced \
#       --num-images 1000 \
#       --num-workers 4 \
#       --worker-id "$i" \
#       --seed 42 &
#   done
#
#   wait
#   python merge.py
#   python draw_bbox.py --limit 10
#   python validate_dataset.py
#
#
# 10) WINDOWS POWERSHELL — 4 PARALEL WORKER
# ------------------------------------------------
# Önce BlenderProc çalıştırılabilir dosyasını bul:
#
#   $BlenderProc = (Get-Command blenderproc).Source
#
# Dört worker başlat:
#
#   $jobs = 0..3 | ForEach-Object {
#       $worker = $_
#       Start-Process -FilePath $BlenderProc -PassThru -ArgumentList @(
#           "run", ".\generate.py", "--",
#           "--optix",
#           "--quality-profile", "balanced",
#           "--num-images", "1000",
#           "--num-workers", "4",
#           "--worker-id", "$worker",
#           "--seed", "42"
#       )
#   }
#
# Worker işlemlerinin tamamlanmasını bekle:
#
#   $jobs | Wait-Process
#
# Çıktıları birleştir ve doğrula:
#
#   python .\merge.py
#   python .\draw_bbox.py --limit 10
#   python .\validate_dataset.py
#
#
# 11) KALİTE PROFİLLERİ
# ------------------------------------------------
#
#   --quality-profile fast
#       32 sample, hızlı üretim
#
#   --quality-profile balanced
#       64 sample, varsayılan ve önerilen profil
#
#   --quality-profile realistic
#       192 sample, daha kaliteli ve yavaş
#
# --samples verilirse profilin sample değeri değiştirilir:
#
#   blenderproc run generate.py -- --optix \
#     --quality-profile balanced \
#     --samples 96 \
#     --num-images 1000
#
#
# 12) GÜNCEL CLI PARAMETRELERİ
# ------------------------------------------------
#
#   --test
#       10 görüntülük hızlı test
#
#   --quality-profile fast|balanced|realistic
#       Render kalite profili
#
#   --optix
#       NVIDIA OptiX GPU
#
#   --gpu
#       NVIDIA CUDA GPU
#
#   --hybrid
#       CUDA CPU+GPU hibrit kullanım
#
#   --eevee
#       Eevee render motoru
#
#   --num-images N
#       Toplam üretilecek görüntü sayısı
#
#   --samples N
#       Cycles sample sayısı
#
#   --seed N
#       Tekrar üretilebilirlik seed'i; varsayılan 42
#
#   --worker-id N
#       Worker numarası; 0'dan başlar
#
#   --num-workers N
#       Toplam worker sayısı
#
#   --texture-pool-size N
#       FDM texture havuzu boyutu; varsayılan 200
#
#   --regen-textures
#       FDM texture havuzunu silip yeniden üretir
#
#   --gen-textures
#       Yalnızca FDM texture üretir, render yapmaz
#
#   --low-res-probability P
#       Düşük çözünürlük uygulanma olasılığı; varsayılan 0.35
#
#   --low-res-sizes 160,240,320,480
#       Kullanılabilecek düşük çözünürlük boyutları
#
#   --jpeg-compression-probability P
#       JPEG artefakt olasılığı; varsayılan 0.25
#
#   --jpeg-quality-range 58,90
#       JPEG kalite aralığı
#
#
# 13) ÇIKTI KLASÖRLERİ
# ------------------------------------------------
#
# Tek worker çıktıları:
#
#   output/yolo/images/
#   output/yolo/labels/
#   output/yolo/metadata/
#   output/coco/images/
#   output/coco/coco_annotations.json
#   output/dataset.yaml
#
# Paralel worker çıktıları:
#
#   output/workers/worker_0/
#   output/workers/worker_1/
#   output/workers/worker_2/
#   output/workers/worker_3/
#
# Paralel üretim bittikten sonra mutlaka çalıştır:
#
#   python merge.py
#
#
# 14) SON KONTROL
# ------------------------------------------------
#
# Syntax ve Blender gerektirmeyen testler:
#
#   python -m compileall -q .
#   python -m unittest discover -s tests -v
#
# Üretilmiş dataset kontrolü:
#
#   python validate_dataset.py
#
# Bounding box önizlemeleri:
#
#   python draw_bbox.py --limit 10
#
# ================================================================
# blenderproc run generate.py -- --optix --quality-profile balanced --num-images 4000 --seed 42