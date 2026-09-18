#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDM texture havuzunu önceden üretir.
BlenderProc gerektirmez — sadece PIL ve numpy kullanır.

Kullanım:
    python gen_textures.py                          # 200 texture, cache varsa atla
    python gen_textures.py --pool-size 200          # 200 texture üret
    python gen_textures.py --regen                  # eski havuzu sil ve yeniden üret
    python gen_textures.py --pool-size 200 --regen  # 200 texture, sıfırdan
"""

import json
import os
import shutil
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── Varsayılan config (generate.py ile senkron) ──────────────────
TEXTURE_POOL_SIZE   = 200
TEXTURE_POOL_DIR    = os.path.join(BASE_DIR, "fdm_textures")
TEXTURE_SIZE        = 512
TEXTURE_MASTER_SEED = 7

# ── CLI argümanları ───────────────────────────────────────────────
def _get_int_arg(flag, default):
    try:
        return int(sys.argv[sys.argv.index(flag) + 1])
    except (ValueError, IndexError):
        return default

_regen    = "--regen"      in sys.argv
_pool_size = _get_int_arg("--pool-size", TEXTURE_POOL_SIZE)
_size      = _get_int_arg("--size",      TEXTURE_SIZE)
_seed      = _get_int_arg("--seed",      TEXTURE_MASTER_SEED)


def generate_texture_pool(pool_size, size, master_seed, pool_dir):
    """
    texture.py kullanarak FDM albedo+normal çiftlerini üretir.
    Mevcut cache yeterli büyüklükteyse yeniden üretmez.
    """
    from texture import (
        generate_variant, MATERIAL_PROFILES,
        SURFACE_PATTERNS, SURFACE_PATTERN_WEIGHTS,
    )

    os.makedirs(pool_dir, exist_ok=True)
    meta_path = os.path.join(pool_dir, "pool_meta.json")

    refresh_existing = False
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            pool = json.load(f)
        if len(pool) >= pool_size and all(
            item.get("pattern_type") and item.get("seed") is not None
            for item in pool[:pool_size]
        ):
            print(f"Cache geçerli: {len(pool)} doku mevcut → {pool_dir}")
            print("Yeniden üretmek için: python gen_textures.py --regen")
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
    print(f"FDM texture havuzu üretiliyor: {pool_size} varyasyon")
    print(f"Hedef dizin : {pool_dir}")
    print(f"Çözünürlük  : {size}×{size}  |  Seed: {master_seed}")
    t0 = time.time()

    for i, (mat, col, rgb, prof) in enumerate(selected):
        seed = int(rng.integers(0, 999_999))
        vid  = f"{i:03d}_{mat}_{col}"
        pattern_type = str(rng.choice(SURFACE_PATTERNS, p=SURFACE_PATTERN_WEIGHTS))
        secondary_rgb = all_variants[int(rng.integers(0, len(all_variants)))][2]

        albedo_path = os.path.join(pool_dir, f"{vid}_albedo.png")
        normal_path = os.path.join(pool_dir, f"{vid}_normal.png")

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
            "base_rgb":  list(rgb),
            "secondary_rgb": variant_meta["secondary_rgb"],
            "pattern_type": variant_meta["pattern_type"],
            "seed": seed,
        })

        if (i + 1) % 10 == 0 or (i + 1) == pool_size:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (pool_size - i - 1)
            print(f"  {i + 1:>4}/{pool_size}  —  geçen {elapsed:.0f}s  beklenen {eta:.0f}s")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)

    total_time = time.time() - t0
    print(f"\nHazır: {len(pool)} doku  ({total_time:.1f}s)  →  {pool_dir}")
    return pool


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Kullanım: python gen_textures.py [--pool-size N] [--size N] "
            "[--seed N] [--regen]"
        )
        return
    if _regen and os.path.isdir(TEXTURE_POOL_DIR):
        shutil.rmtree(TEXTURE_POOL_DIR)
        print(f"Eski texture havuzu silindi: {TEXTURE_POOL_DIR}")

    generate_texture_pool(_pool_size, _size, _seed, TEXTURE_POOL_DIR)


if __name__ == "__main__":
    main()
