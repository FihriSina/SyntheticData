#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO etiket dosyalarındaki bbox'ları görüntüler üzerine çizer.
Çıktılar output/bbox_debug/ klasörüne kaydedilir.

Kullanım:
    python draw_bbox.py
"""

import os
import glob
import sys
from PIL import Image, ImageDraw

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

YOLO_IMAGES_DIR = os.path.join(OUTPUT_DIR, "yolo", "images")
YOLO_LABELS_DIR = os.path.join(OUTPUT_DIR, "yolo", "labels")
BBOX_DEBUG_DIR  = os.path.join(OUTPUT_DIR, "bbox_debug")

BBOX_LINE_WIDTH  = 3
BBOX_COLOR       = (0, 255, 0)
BBOX_LABEL_TEXT  = "cube"
BBOX_LABEL_COLOR = (255, 255, 0)


def _get_int_arg(flag, default):
    try:
        return int(sys.argv[sys.argv.index(flag) + 1])
    except (ValueError, IndexError):
        return default


BBOX_LIMIT = _get_int_arg("--limit", 0)  # 0 = tüm görüntüler


def draw_bboxes():
    os.makedirs(BBOX_DEBUG_DIR, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(YOLO_IMAGES_DIR, "**", "*.jpg"), recursive=True) +
        glob.glob(os.path.join(YOLO_IMAGES_DIR, "**", "*.jpeg"), recursive=True) +
        glob.glob(os.path.join(YOLO_IMAGES_DIR, "**", "*.png"), recursive=True)
    )
    if BBOX_LIMIT > 0:
        image_paths = image_paths[:BBOX_LIMIT]

    if not image_paths:
        print(f"Goruntu bulunamadi: {YOLO_IMAGES_DIR}")
        return

    processed = 0
    for img_path in image_paths:
        relative = os.path.relpath(img_path, YOLO_IMAGES_DIR)
        stem_rel = os.path.splitext(relative)[0]
        lbl_path = os.path.join(YOLO_LABELS_DIR, stem_rel + ".txt")

        if not os.path.isfile(lbl_path):
            continue

        img  = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        iw, ih = img.size

        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x_min = (cx - bw / 2) * iw
                y_min = (cy - bh / 2) * ih
                x_max = (cx + bw / 2) * iw
                y_max = (cy + bh / 2) * ih
                draw.rectangle(
                    [x_min, y_min, x_max, y_max],
                    outline=BBOX_COLOR,
                    width=BBOX_LINE_WIDTH,
                )
                draw.text((x_min + 2, y_min + 2), BBOX_LABEL_TEXT, fill=BBOX_LABEL_COLOR)

        debug_path = os.path.join(BBOX_DEBUG_DIR, relative)
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        img.save(debug_path)
        processed += 1

    print(f"Bbox gorselleri kaydedildi: {processed} goruntu -> {BBOX_DEBUG_DIR}")


if __name__ == "__main__":
    draw_bboxes()
