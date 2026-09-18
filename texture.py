"""
FDM 3D Print Texture + Normal Map Generator  v4
================================================
Her varyasyon için üretir:
  - albedo.png   : Gerçekçi renkli diffuse texture
  - normalmap.png: OpenGL convention normal map (R=X G=Y B=Z)
"""

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, convolve1d
import os, json, time

# ═══════════════════════════════════════════════════════════
#  GENEL AYARLAR
# ═══════════════════════════════════════════════════════════
SIZE       = 512
N_VARIANTS = 8
OUT_DIR    = "fdm_v3"

# ─── FBM GÜRÜLTÜ AYARLARI ──────────────────────────────────
FBM_GAUSSIAN_SIGMA      = 1.0   # Her oktav öncesi uygulanan blur

# ─── WOBBLE (KATMAN DALGALANMA) AYARLARI ───────────────────
WOBBLE_LONG_OCTAVES     = 3     # Uzun dalga FBM oktav sayısı
WOBBLE_LONG_PERSISTENCE = 0.7   # Uzun dalga kalıcılığı
WOBBLE_LONG_BASE_SCALE  = 96    # Uzun dalga baz ölçeği (px)

WOBBLE_SHORT_OCTAVES    = 5     # Kısa dalga FBM oktav sayısı
WOBBLE_SHORT_PERSISTENCE= 0.55  # Kısa dalga kalıcılığı
WOBBLE_SHORT_BASE_SCALE = 28    # Kısa dalga baz ölçeği (px)

WOBBLE_XDRIFT_OCTAVES   = 3     # X-kayma FBM oktav sayısı
WOBBLE_XDRIFT_PERSIST   = 0.6   # X-kayma kalıcılığı
WOBBLE_XDRIFT_SCALE     = 48    # X-kayma baz ölçeği (px)
WOBBLE_XDRIFT_WEIGHT    = 0.40  # X-kayma amplitüd çarpanı

WOBBLE_LONG_WEIGHT      = 0.65  # Uzun dalganın birleşim ağırlığı
WOBBLE_SHORT_WEIGHT     = 0.35  # Kısa dalganın birleşim ağırlığı

# ─── BEAD (KATMAN KESİTİ) AYARLARI ────────────────────────
BEAD_EXPONENT           = 0.6   # Bead profil eğrisi (düşük=yayvan, yüksek=sivri)
EDGE_SHARP_FACTOR       = 0.7   # Kenar gölge bölgesi daraltıcı (< 1 = yumuşak)
EDGE_SHADOW_EXPONENT    = 0.9   # Kenar gölge geçiş eğrisi (düşük=yumuşak)

# ─── EXTRUSION VARYASYONU AYARLARI ────────────────────────
EXT_BASE                = 0.78  # Minimum extrusion oranı
EXT_VARIATION           = 0.22  # Maksimum ek extrusion (beta dağılımı)
EXT_BETA_A              = 9     # Beta dağılımı α parametresi
EXT_BETA_B              = 2     # Beta dağılımı β parametresi
EXT_XWAVE_AMP1          = 0.035 # X yönü 1. dalga amplitüdü
EXT_XWAVE_FREQ1         = 3.7   # X yönü 1. dalga frekansı
EXT_XWAVE_AMP2          = 0.018 # X yönü 2. dalga amplitüdü
EXT_XWAVE_FREQ2         = 11.2  # X yönü 2. dalga frekansı
EXT_FBM_OCTAVES         = 4     # Extrusion FBM oktav sayısı
EXT_FBM_BASE_SCALE      = 60    # Extrusion FBM baz ölçeği (px)
EXT_NOISE_WEIGHT        = 0.15  # Extrusion gürültü katkı ağırlığı

# ─── THERMAL GHOSTING AYARLARI ────────────────────────────
THERM_LOW_OCTAVES       = 3     # Düşük frekans FBM oktavları
THERM_LOW_PERSIST       = 0.6   # Düşük frekans kalıcılığı
THERM_LOW_BASE_SCALE    = 120   # Düşük frekans baz ölçeği (px)
THERM_MID_OCTAVES       = 2     # Orta frekans FBM oktavları
THERM_MID_PERSIST       = 0.4   # Orta frekans kalıcılığı
THERM_MID_BASE_SCALE    = 40    # Orta frekans baz ölçeği (px)
THERM_LOW_WEIGHT        = 0.7   # Düşük frekans birleşim ağırlığı
THERM_MID_WEIGHT        = 0.3   # Orta frekans birleşim ağırlığı
THERM_STRENGTH          = 0.04  # Termal etki gücü (artır → daha fazla koyu yama)

# ─── RINGING (MEKANİK TİTREŞİM) AYARLARI ─────────────────
RING_F1_MIN             = 0.18  # 1. titreşim frekansı minimum
RING_F1_MAX             = 0.25  # 1. titreşim frekansı maksimum
RING_F2_MIN             = 0.35  # 2. titreşim frekansı minimum
RING_F2_MAX             = 0.45  # 2. titreşim frekansı maksimum
RING_DECAY              = 0.015 # Titreşim sönümlenme katsayısı
RING_AMP                = 0.025 # Titreşim genliği
RING_W1                 = 0.6   # 1. bileşen ağırlığı
RING_W2                 = 0.4   # 2. bileşen ağırlığı
RING_PHASE_OFFSET       = 1.3   # 2. bileşen faz kaydırması (rad)

# ─── PHONG IŞIKLANDIRMA AYARLARI ──────────────────────────
PHONG_LX                = 0.4   # Işık yönü X bileşeni
PHONG_LY                = -0.3  # Işık yönü Y bileşeni
PHONG_LZ                = 0.9   # Işık yönü Z bileşeni
PHONG_AMBIENT           = 0.35  # Ortam ışığı miktarı
PHONG_DIFFUSE_W         = 0.50  # Difüz katkı ağırlığı
PHONG_SPEC_STR_MATTE    = 0.12  # Mat yüzey specular gücü
PHONG_SPEC_STR_SHINY    = 0.40  # Parlak yüzey specular gücü
PHONG_SPEC_THRESHOLD    = 20    # spec_power eşiği (üstü = parlak)

# ─── MİKRO ÇİZİK AYARLARI ─────────────────────────────────
SCRATCH_KERNEL_SIZE     = 11    # Yatay bulanıklaştırma kernel genişliği
SCRATCH_SIGMA_X         = 0.3   # Y yönü Gaussian sigma
SCRATCH_SIGMA_Y         = 0.0   # X yönü Gaussian sigma (0 = anizotropik)
SCRATCH_WEIGHT          = 0.04  # Çizik katkı ağırlığı

# ─── YÜZEY PÜRÜZLÜLÜĞİ AYARLARI ─────────────────────────
SURF_FBM_OCTAVES        = 7     # Yüzey FBM oktav sayısı
SURF_FBM_PERSISTENCE    = 0.45  # Yüzey FBM kalıcılığı
SURF_FBM_BASE_SCALE     = 8     # Yüzey FBM baz ölçeği (px)
SURF_WEIGHT             = 0.06  # Yüzey pürüzlülük katkı ağırlığı

# ─── VİGNETTE AYARLARI ────────────────────────────────────
VIGNETTE_STRENGTH       = 0.14  # Köşe kararma gücü (0 = yok)
VIGNETTE_EXPONENT       = 1.8   # Kararma eğrisi

# ─── HEIGHTMAP NORMALİZASYON AYARLARI ─────────────────────
HEIGHT_PERCENTILE_LO    = 1     # Alt normalize yüzdelik
HEIGHT_PERCENTILE_HI    = 99    # Üst normalize yüzdelik
HEIGHT_BLUR_RADIUS      = 0.55  # Son diffüzyon bulanıklığı (px)

# ─── NORMAL MAP AYARLARI ──────────────────────────────────
NORMALMAP_STRENGTH_MULT = 8.0   # Normal harita güç çarpanı (roughness ile çarpılır)

# ─── ALBEDO AYARLARI ──────────────────────────────────────
ALBEDO_SHADOW_FLOOR     = 0.52  # Minimum parlaklık tabanı (artır → az siyah leke)
ALBEDO_SHADOW_RANGE     = 0.62  # Parlaklık dinamik aralığı (FLOOR + RANGE = 1.0)
ALBEDO_CN_OCTAVES       = 4     # Pigment varyasyonu FBM oktavları
ALBEDO_CN_BASE_SCALE    = 80    # Pigment varyasyonu FBM baz ölçeği (px)
ALBEDO_CN_WEIGHT        = 0.025 # Pigment gürültü katkısı (artır → daha fazla leke)
ALBEDO_R_SHIFT          = 0.04  # Kırmızı kanal maksimum rastgele kayması
ALBEDO_G_SHIFT          = 0.03  # Yeşil kanal maksimum rastgele kayması
ALBEDO_B_SHIFT          = 0.03  # Mavi kanal maksimum rastgele kayması
ALBEDO_BLUR_RADIUS      = 0.4   # Son plastik difüzyon bulanıklığı (px)

# ─── YÜZEY DESENLERİ ───────────────────────────────────────────────────────
SURFACE_PATTERNS = [
    "striped", "grid", "dotted", "geometric", "speckled", "marble",
    "two_tone", "gradient", "worn", "pronounced_layers",
    "filament_wobble", "print_defect",
]
SURFACE_PATTERN_WEIGHTS = [0.10, 0.08, 0.08, 0.08, 0.10, 0.10,
                           0.08, 0.08, 0.08, 0.08, 0.07, 0.07]


# ─── MATERYAL PROFİLLERİ ────────────────────────────────────────────────────
MATERIAL_PROFILES = {
    "PLA_MATTE": dict(
        lt_range=(10, 18), noise_range=(0.18, 0.30),
        roughness=0.55, spec_power=12,
        wobble_range=(4.0, 8.0), edge_sharp=0.07,
        colors=[
            ("white",    (235, 232, 228)),
            ("black",    ( 28,  26,  25)),
            ("red",      (195,  42,  38)),
            ("blue",     ( 38,  82, 180)),
            ("green",    ( 42, 148,  60)),
            ("yellow",   (220, 195,  35)),
            ("orange",   (215, 105,  30)),
            ("gray",     (130, 128, 126)),
        ],
    ),
    "PLA_SILK": dict(
        lt_range=(8, 14), noise_range=(0.08, 0.16),
        roughness=0.25, spec_power=45,
        wobble_range=(2.5, 5.0), edge_sharp=0.05,
        colors=[
            ("gold",        (210, 168,  60)),
            ("silver",      (185, 188, 192)),
            ("copper",      (185,  95,  55)),
            ("rose_gold",   (210, 140, 120)),
            ("pearl_white", (238, 235, 225)),
            ("deep_blue",   ( 25,  55, 140)),
        ],
    ),
    "PETG": dict(
        lt_range=(12, 20), noise_range=(0.22, 0.38),
        roughness=0.65, spec_power=28,
        wobble_range=(5.0, 10.0), edge_sharp=0.09,
        colors=[
            ("natural",     (210, 205, 195)),
            ("trans_blue",  ( 60, 130, 200)),
            ("trans_red",   (200,  65,  55)),
            ("trans_green", ( 55, 175,  80)),
            ("black",       ( 22,  20,  20)),
            ("white",       (228, 225, 220)),
        ],
    ),
    "ABS": dict(
        lt_range=(14, 22), noise_range=(0.28, 0.45),
        roughness=0.80, spec_power=8,
        wobble_range=(6.0, 12.0), edge_sharp=0.10,
        colors=[
            ("white",  (225, 222, 215)),
            ("black",  ( 20,  18,  18)),
            ("gray",   (105, 102,  98)),
            ("red",    (188,  35,  30)),
            ("ivory",  (220, 208, 180)),
        ],
    ),
    "TPU": dict(
        lt_range=(10, 18), noise_range=(0.32, 0.50),
        roughness=0.70, spec_power=18,
        wobble_range=(6.0, 12.0), edge_sharp=0.12,
        colors=[
            ("black", ( 18,  16,  15)),
            ("white", (220, 215, 208)),
            ("clear", (195, 192, 185)),
            ("red",   (175,  30,  25)),
            ("blue",  ( 30,  75, 165)),
        ],
    ),
    "PLA_METALLIC": dict(
        lt_range=(8, 14), noise_range=(0.06, 0.14),
        roughness=0.20, spec_power=80, metallic=0.85,
        wobble_range=(2.0, 4.5), edge_sharp=0.04,
        colors=[
            ("silver",      (190, 192, 196)),
            ("gold",        (200, 162,  50)),
            ("bronze",      (170, 110,  60)),
            ("chrome",      (210, 212, 215)),
            ("rose_gold",   (205, 140, 115)),
            ("gunmetal",    ( 80,  85,  90)),
        ],
    ),
    "PLA_GLOSSY": dict(
        lt_range=(7, 13), noise_range=(0.04, 0.10),
        roughness=0.08, spec_power=120, metallic=0.0,
        wobble_range=(1.5, 3.5), edge_sharp=0.03,
        colors=[
            ("gloss_white",  (245, 243, 240)),
            ("gloss_black",  ( 15,  13,  12)),
            ("gloss_red",    (210,  30,  25)),
            ("gloss_blue",   ( 20,  60, 200)),
            ("gloss_yellow", (230, 210,  20)),
            ("gloss_green",  ( 20, 160,  50)),
        ],
    ),
    "WOOD_PLA": dict(
        lt_range=(12, 20), noise_range=(0.40, 0.65),
        roughness=0.85, spec_power=5, metallic=0.0,
        wobble_range=(7.0, 14.0), edge_sharp=0.12,
        colors=[
            ("oak",      (185, 140,  80)),
            ("walnut",   (110,  65,  30)),
            ("pine",     (215, 185, 120)),
            ("ebony",    ( 35,  22,  12)),
        ],
    ),
    "MARBLE_PLA": dict(
        lt_range=(9, 16), noise_range=(0.15, 0.28),
        roughness=0.30, spec_power=60, metallic=0.0,
        wobble_range=(3.5, 7.0), edge_sharp=0.06,
        colors=[
            ("white_marble",  (235, 232, 228)),
            ("black_marble",  ( 30,  28,  26)),
            ("green_marble",  (100, 140, 110)),
            ("pink_marble",   (215, 185, 180)),
        ],
    ),
}


# ═══════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════

def fbm_noise(shape, rng, octaves=6, persistence=0.5, lacunarity=2.0, base_scale=32):
    """Çok oktavlı organik gürültü. Döner: float32 [-1, 1]"""
    h, w = shape
    result, amp, freq, total = np.zeros((h, w), np.float32), 1.0, 1.0, 0.0
    for _ in range(octaves):
        sc = max(int(base_scale / freq), 2)
        sh = max(2, h // sc)
        sw = max(2, w // sc)
        small = rng.standard_normal((sh, sw)).astype(np.float32)
        small = gaussian_filter(small, sigma=FBM_GAUSSIAN_SIGMA)
        big = np.array(Image.fromarray(small, mode="F").resize((w, h), Image.BILINEAR))
        result += amp * big
        total  += amp
        amp    *= persistence
        freq   *= lacunarity
    return result / total


def make_heightmap(size, lt, wobble_amp, edge_sharp, noise_level, spec_power, rng):
    """
    Tüm FDM fizik efektlerini birleştirip normalize heightmap döner.
    Döner: float32 [0, 1]
    """
    h = w = size
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))

    # 1. Z-wobble → 2D FBM displacement (her X konumunda bağımsız bükülme)
    long_disp = fbm_noise((h, w), rng,
                          octaves=WOBBLE_LONG_OCTAVES,
                          persistence=WOBBLE_LONG_PERSISTENCE,
                          base_scale=WOBBLE_LONG_BASE_SCALE)
    long_disp = long_disp / (np.std(long_disp) + 1e-8)

    short_disp = fbm_noise((h, w), rng,
                           octaves=WOBBLE_SHORT_OCTAVES,
                           persistence=WOBBLE_SHORT_PERSISTENCE,
                           base_scale=WOBBLE_SHORT_BASE_SCALE)
    short_disp = short_disp / (np.std(short_disp) + 1e-8)

    x_drift = fbm_noise((h, w), rng,
                        octaves=WOBBLE_XDRIFT_OCTAVES,
                        persistence=WOBBLE_XDRIFT_PERSIST,
                        base_scale=WOBBLE_XDRIFT_SCALE)
    x_drift = x_drift / (np.std(x_drift) + 1e-8) * wobble_amp * WOBBLE_XDRIFT_WEIGHT

    wobble = wobble_amp * (WOBBLE_LONG_WEIGHT * long_disp +
                           WOBBLE_SHORT_WEIGHT * short_disp) + x_drift
    yy_w = yy + wobble

    layer_idx = np.floor(yy_w / lt).astype(np.int32)
    layer_pos = (yy_w % lt) / lt

    # 2. Bead cross-section profili
    bead = np.sin(np.pi * layer_pos) ** BEAD_EXPONENT

    # 3. Katman kenar gölgesi
    ew = edge_sharp * EDGE_SHARP_FACTOR
    shadow = np.where(
        layer_pos < ew,
        (layer_pos / ew) ** EDGE_SHADOW_EXPONENT,
        np.where(layer_pos > 1 - ew,
                 ((1 - layer_pos) / ew) ** EDGE_SHADOW_EXPONENT,
                 1.0)
    )
    bead = bead * shadow

    # 4. Extrusion varyasyonu
    num_layers = h // lt + 2
    per_layer = (EXT_BASE +
                 EXT_VARIATION * rng.beta(EXT_BETA_A, EXT_BETA_B,
                                          size=num_layers).astype(np.float32))
    ext_layer = per_layer[np.clip(layer_idx, 0, num_layers - 1)]
    x_wave = (1.0
              + EXT_XWAVE_AMP1 * np.sin(2 * np.pi * EXT_XWAVE_FREQ1 * xx / w)
              + EXT_XWAVE_AMP2 * np.sin(2 * np.pi * EXT_XWAVE_FREQ2 * xx / w))
    ext_map = ext_layer * x_wave
    ext_map += (fbm_noise((h, w), rng,
                          octaves=EXT_FBM_OCTAVES,
                          base_scale=EXT_FBM_BASE_SCALE)
                * noise_level * EXT_NOISE_WEIGHT)

    # 5. Thermal ghosting
    low = fbm_noise((h, w), rng,
                    octaves=THERM_LOW_OCTAVES,
                    persistence=THERM_LOW_PERSIST,
                    base_scale=THERM_LOW_BASE_SCALE)
    mid = fbm_noise((h, w), rng,
                    octaves=THERM_MID_OCTAVES,
                    persistence=THERM_MID_PERSIST,
                    base_scale=THERM_MID_BASE_SCALE)
    therm = THERM_LOW_WEIGHT * low + THERM_MID_WEIGHT * mid
    therm = 1.0 + noise_level * THERM_STRENGTH * therm / (np.std(therm) + 1e-8)

    # 6. Ringing (mekanik titreşim izi)
    f1 = rng.uniform(RING_F1_MIN, RING_F1_MAX)
    f2 = rng.uniform(RING_F2_MIN, RING_F2_MAX)
    decay = np.exp(-RING_DECAY * yy)
    ring = 1.0 + RING_AMP * (RING_W1 * np.sin(2 * np.pi * f1 * yy) +
                               RING_W2 * np.sin(2 * np.pi * f2 * yy +
                                                 RING_PHASE_OFFSET)) * decay

    # 7. Phong lighting
    dz_dy = np.gradient(bead, axis=0)
    dz_dx = np.gradient(bead, axis=1)
    Lx, Ly, Lz = PHONG_LX, PHONG_LY, PHONG_LZ
    L = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    Lx, Ly, Lz = Lx / L, Ly / L, Lz / L
    nx, ny, nz = -dz_dx, -dz_dy, np.ones_like(dz_dx)
    n = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx / n, ny / n, nz / n
    diffuse = np.clip(nx * Lx + ny * Ly + nz * Lz, 0, 1)
    spec_str = (PHONG_SPEC_STR_SHINY if spec_power >= PHONG_SPEC_THRESHOLD
                else PHONG_SPEC_STR_MATTE)
    specular = np.clip(nz, 0, 1) ** spec_power * spec_str
    phong = PHONG_AMBIENT + PHONG_DIFFUSE_W * diffuse + specular

    # 8. Anizotropik mikro çiziklar
    scratch = rng.standard_normal((h, w)).astype(np.float32)
    scratch = convolve1d(scratch, np.ones(SCRATCH_KERNEL_SIZE) / SCRATCH_KERNEL_SIZE,
                         axis=1)
    scratch = gaussian_filter(scratch, sigma=[SCRATCH_SIGMA_X, SCRATCH_SIGMA_Y])
    scratch = noise_level * SCRATCH_WEIGHT * scratch / (np.std(scratch) + 1e-8)

    # 9. Organik yüzey pürüzlülüğü
    surf = fbm_noise((h, w), rng,
                     octaves=SURF_FBM_OCTAVES,
                     persistence=SURF_FBM_PERSISTENCE,
                     base_scale=SURF_FBM_BASE_SCALE)
    surf = noise_level * SURF_WEIGHT * surf / (np.std(surf) + 1e-8)

    # 10. Vignette
    cy, cx = h / 2, w / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vig = 1.0 - VIGNETTE_STRENGTH * np.clip(dist, 0, 1) ** VIGNETTE_EXPONENT

    # Birleştir
    texture = (phong * ext_map * therm * ring + scratch + surf) * vig

    # Normalize → [0, 1]
    lo, hi = np.percentile(texture, [HEIGHT_PERCENTILE_LO, HEIGHT_PERCENTILE_HI])
    height = np.clip((texture - lo) / (hi - lo + 1e-8), 0, 1)

    # Hafif blur (plastik difüzyon)
    height_img = Image.fromarray((height * 255).astype(np.uint8), mode="L")
    height_img = height_img.filter(ImageFilter.GaussianBlur(radius=HEIGHT_BLUR_RADIUS))
    return np.array(height_img).astype(np.float32) / 255.0


def height_to_normalmap(height, roughness):
    """float32 [0,1] heightmap → OpenGL normal map RGB uint8"""
    strength = roughness * NORMALMAP_STRENGTH_MULT
    dz_dx = np.gradient(height, axis=1) * strength
    dz_dy = np.gradient(height, axis=0) * strength

    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(nx)
    length = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx, ny, nz = nx / length, ny / length, nz / length

    r = np.clip((nx * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    g = np.clip((ny * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    b = np.clip((nz * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def height_to_albedo(height, base_rgb, noise_level, rng):
    """height: float32 [0,1], base_rgb: (R,G,B) int 0-255 → PIL RGB image"""
    R, G, B = [c / 255.0 for c in base_rgb]

    cn = fbm_noise(height.shape, rng,
                   octaves=ALBEDO_CN_OCTAVES,
                   base_scale=ALBEDO_CN_BASE_SCALE)
    cn = cn / (np.std(cn) + 1e-8) * noise_level * ALBEDO_CN_WEIGHT

    height_lit = ALBEDO_SHADOW_FLOOR + ALBEDO_SHADOW_RANGE * height

    r_shift = rng.uniform(-ALBEDO_R_SHIFT, ALBEDO_R_SHIFT)
    g_shift = rng.uniform(-ALBEDO_G_SHIFT, ALBEDO_G_SHIFT)
    b_shift = rng.uniform(-ALBEDO_B_SHIFT, ALBEDO_B_SHIFT)

    ch_r = np.clip(height_lit * (R + r_shift) + cn * R, 0, 1)
    ch_g = np.clip(height_lit * (G + g_shift) + cn * G, 0, 1)
    ch_b = np.clip(height_lit * (B + b_shift) + cn * B, 0, 1)

    arr = np.stack([
        (ch_r * 255).astype(np.uint8),
        (ch_g * 255).astype(np.uint8),
        (ch_b * 255).astype(np.uint8),
    ], axis=-1)

    img = Image.fromarray(arr, mode="RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=ALBEDO_BLUR_RADIUS))


def apply_surface_pattern(albedo, height, pattern_type, secondary_rgb, rng):
    """FDM albedo/height çiftine kontrollü bir yüzey deseni uygular."""
    arr = np.asarray(albedo).astype(np.float32) / 255.0
    out_height = np.asarray(height, dtype=np.float32).copy()
    h, w = out_height.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    secondary = np.asarray(secondary_rgb, dtype=np.float32) / 255.0
    secondary = secondary.reshape(1, 1, 3)
    strength = float(rng.uniform(0.16, 0.32))

    if pattern_type == "striped":
        angle = float(rng.uniform(0, np.pi))
        coord = xx * np.cos(angle) + yy * np.sin(angle)
        mask = (np.sin(coord / rng.uniform(8.0, 18.0)) > 0).astype(np.float32)
        mask = gaussian_filter(mask, sigma=0.7)[..., None]
        arr = arr * (1.0 - strength * mask) + secondary * strength * mask

    elif pattern_type == "grid":
        spacing = int(rng.integers(max(18, w // 18), max(22, w // 9)))
        thickness = int(rng.integers(1, 4))
        mask = ((xx.astype(int) % spacing < thickness) |
                (yy.astype(int) % spacing < thickness)).astype(np.float32)[..., None]
        arr = arr * (1.0 - strength * mask) + secondary * strength * mask

    elif pattern_type == "dotted":
        spacing = int(rng.integers(max(20, w // 20), max(28, w // 10)))
        radius = max(2, int(spacing * rng.uniform(0.10, 0.22)))
        dx = (xx.astype(int) % spacing) - spacing / 2
        dy = (yy.astype(int) % spacing) - spacing / 2
        mask = ((dx * dx + dy * dy) <= radius * radius).astype(np.float32)[..., None]
        arr = arr * (1.0 - strength * mask) + secondary * strength * mask

    elif pattern_type == "geometric":
        cell = int(rng.integers(max(28, w // 14), max(40, w // 7)))
        tri = ((xx.astype(int) % cell) + (yy.astype(int) % cell) > cell).astype(np.float32)
        checker = (((xx.astype(int) // cell) + (yy.astype(int) // cell)) % 2).astype(np.float32)
        mask = np.logical_xor(tri > 0, checker > 0).astype(np.float32)[..., None]
        arr = arr * (1.0 - strength * mask) + secondary * strength * mask

    elif pattern_type == "speckled":
        sparse = (rng.random((h, w)) > 0.985).astype(np.float32)
        mask = gaussian_filter(sparse, sigma=float(rng.uniform(0.6, 1.5)))
        mask = np.clip(mask * 4.0, 0.0, 1.0)[..., None]
        arr = arr * (1.0 - strength * mask) + secondary * strength * mask

    elif pattern_type == "marble":
        organic = fbm_noise((h, w), rng, octaves=5, persistence=0.58,
                            base_scale=max(20, w // 5))
        coord = xx / rng.uniform(18.0, 35.0) + organic * rng.uniform(3.0, 7.0)
        veins = np.exp(-np.abs(np.sin(coord)) * rng.uniform(8.0, 14.0))
        mask = gaussian_filter(veins, sigma=0.8)[..., None]
        arr = arr * (1.0 - strength * mask) + secondary * strength * mask

    elif pattern_type == "two_tone":
        angle = float(rng.uniform(0, np.pi))
        plane = (xx - w / 2) * np.cos(angle) + (yy - h / 2) * np.sin(angle)
        mask = gaussian_filter((plane > 0).astype(np.float32), sigma=1.2)[..., None]
        arr = arr * (1.0 - 0.58 * mask) + secondary * 0.58 * mask

    elif pattern_type == "gradient":
        angle = float(rng.uniform(0, np.pi))
        grad = ((xx / max(w - 1, 1)) * np.cos(angle) +
                (yy / max(h - 1, 1)) * np.sin(angle))
        grad = (grad - grad.min()) / (np.ptp(grad) + 1e-8)
        mask = (grad * rng.uniform(0.35, 0.65))[..., None]
        arr = arr * (1.0 - mask) + secondary * mask

    elif pattern_type == "worn":
        wear = fbm_noise((h, w), rng, octaves=5, persistence=0.52,
                         base_scale=max(14, w // 8))
        wear = np.clip((wear - np.percentile(wear, 58)) * 3.5, 0, 1)[..., None]
        gray = np.mean(arr, axis=2, keepdims=True)
        arr = arr * (1.0 - 0.28 * wear) + gray * 0.28 * wear
        out_height = np.clip(out_height - wear[..., 0] * 0.035, 0, 1)

    elif pattern_type == "pronounced_layers":
        period = float(rng.uniform(7.0, 14.0))
        bands = (0.5 + 0.5 * np.sin(2 * np.pi * yy / period))
        arr *= (0.93 + 0.09 * bands[..., None])
        out_height = np.clip(out_height + (bands - 0.5) * 0.075, 0, 1)

    elif pattern_type == "filament_wobble":
        wobble = np.sin(xx / rng.uniform(22.0, 48.0) +
                        yy / rng.uniform(6.0, 12.0))
        arr *= (0.96 + 0.055 * wobble[..., None])
        out_height = np.clip(out_height + wobble * 0.045, 0, 1)

    elif pattern_type == "print_defect":
        defect = np.zeros((h, w), dtype=np.float32)
        for _ in range(int(rng.integers(1, 4))):
            row = int(rng.integers(4, h - 4))
            thickness = int(rng.integers(1, 4))
            defect[max(0, row - thickness):min(h, row + thickness + 1)] = 1.0
        defect = gaussian_filter(defect, sigma=0.6)
        arr *= (1.0 - defect[..., None] * rng.uniform(0.10, 0.22))
        out_height = np.clip(out_height - defect * 0.09, 0, 1)

    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB"), out_height


# ═══════════════════════════════════════════════════════════
#  TEK VARİASYON ÜRETİCİ
# ═══════════════════════════════════════════════════════════
def generate_variant(mat_name, color_name, base_rgb, profile, seed, size=512,
                     pattern_type=None, secondary_rgb=None):
    rng = np.random.default_rng(seed)

    lt     = int(rng.integers(profile["lt_range"][0], profile["lt_range"][1]))
    noise  = float(rng.uniform(*profile["noise_range"]))
    wobble = float(rng.uniform(*profile["wobble_range"]))

    height = make_heightmap(
        size        = size,
        lt          = lt,
        wobble_amp  = wobble,
        edge_sharp  = profile["edge_sharp"],
        noise_level = noise,
        spec_power  = profile["spec_power"],
        rng         = rng,
    )

    albedo = height_to_albedo(height, base_rgb, noise, rng)
    if pattern_type is None:
        pattern_type = str(rng.choice(SURFACE_PATTERNS, p=SURFACE_PATTERN_WEIGHTS))
    if secondary_rgb is None:
        secondary_rgb = tuple(int(np.clip(255 - c + rng.integers(-24, 25), 0, 255))
                              for c in base_rgb)
    albedo, height = apply_surface_pattern(
        albedo, height, pattern_type, secondary_rgb, rng
    )
    normal_arr = height_to_normalmap(height, profile["roughness"])
    normal_img = Image.fromarray(normal_arr, mode="RGB")

    meta = dict(
        material=mat_name, color=color_name, seed=seed,
        layer_thickness_px=lt, noise_level=round(noise, 3),
        wobble_amplitude=round(wobble, 2),
        roughness=profile["roughness"],
        metallic=profile.get("metallic", 0.0),
        specular_power=profile["spec_power"],
        base_rgb=list(base_rgb),
        secondary_rgb=list(secondary_rgb),
        pattern_type=pattern_type,
    )
    return albedo, normal_img, meta


# ═══════════════════════════════════════════════════════════
#  BATCH ÜRETİM
# ═══════════════════════════════════════════════════════════
def generate_batch(n=N_VARIANTS, size=SIZE, master_seed=42):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng_m    = np.random.default_rng(master_seed)
    all_meta = []

    pool = []
    for mat_name, prof in MATERIAL_PROFILES.items():
        for color_name, rgb in prof["colors"]:
            pool.append((mat_name, color_name, rgb, prof))

    idx      = rng_m.choice(len(pool), size=min(n, len(pool)), replace=False)
    selected = [pool[i] for i in idx]

    print(f"\n🖨️  {len(selected)} varyasyon üretiliyor → {OUT_DIR}\n")
    print(f"  {'#':<4} {'Materyal':<14} {'Renk':<14} {'lt':>4} {'noise':>6}  RGB")
    print("  " + "-" * 60)

    for i, (mat, col, rgb, prof) in enumerate(selected):
        seed = int(rng_m.integers(0, 99999))
        vid  = f"{i+1:02d}_{mat}_{col}"

        t0                = time.time()
        albedo, normal, meta = generate_variant(mat, col, rgb, prof, seed, size)

        albedo_path = f"{OUT_DIR}/{vid}_albedo.png"
        normal_path = f"{OUT_DIR}/{vid}_normalmap.png"
        albedo.save(albedo_path)
        normal.save(normal_path)

        a_arr = np.array(albedo)
        n_arr = np.array(normal)
        print(f"  [{i+1}/{len(selected)}] {mat:<14} {col:<14} "
              f"lt={meta['layer_thickness_px']:>2}  n={meta['noise_level']:.2f}  "
              f"RGB={list(rgb)}  {time.time()-t0:.1f}s")
        print(f"       albedo:  mean={a_arr.mean():.0f} std={a_arr.std():.1f}")
        print(f"       normal:  mean={n_arr.mean():.0f} std={n_arr.std():.1f}")

        meta["files"] = {"albedo": f"{vid}_albedo.png",
                         "normalmap": f"{vid}_normalmap.png"}
        all_meta.append(meta)

    with open(f"{OUT_DIR}/metadata.json", "w") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Tamamlandı → {OUT_DIR}")
    return all_meta


if __name__ == "__main__":
    generate_batch(n=N_VARIANTS)
