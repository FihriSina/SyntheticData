#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Üretilmiş YOLO/COCO/metadata sözleşmesini Blender olmadan doğrular."""

import argparse
import glob
import json
import os
import sys

from PIL import Image


def validate(output_dir):
    errors = []
    warnings = []
    yolo_dir = os.path.join(output_dir, "yolo")
    label_paths = glob.glob(os.path.join(yolo_dir, "labels", "**", "*.txt"), recursive=True)
    image_paths = (
        glob.glob(os.path.join(yolo_dir, "images", "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(yolo_dir, "images", "**", "*.png"), recursive=True)
    )
    metadata_paths = glob.glob(
        os.path.join(yolo_dir, "metadata", "**", "*.json"), recursive=True
    )

    for label_path in label_paths:
        with open(label_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path}:{line_number}: YOLO satırı 5 alan değil")
                    continue
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
                if class_id != 0:
                    errors.append(f"{label_path}:{line_number}: küp dışı class_id={class_id}")
                if not all(0.0 <= value <= 1.0 for value in values):
                    errors.append(f"{label_path}:{line_number}: normalize koordinat aralık dışında")
                if values[2] <= 0.0 or values[3] <= 0.0:
                    errors.append(f"{label_path}:{line_number}: bbox alanı sıfır")

    for image_path in image_paths:
        with Image.open(image_path) as image:
            if image.size != (640, 640):
                errors.append(f"{image_path}: çıktı çözünürlüğü {image.size}, 640x640 bekleniyor")

    low_res_frames = 0
    environment_count = 0
    background_types = {}
    background_files = {}
    for metadata_path in metadata_paths:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        post = metadata.get("post_aug", {})
        background = metadata.get("background", {})
        source_type = background.get("source_type", "unknown")
        source_file = background.get("name", "unknown")
        floor_type = metadata.get("floor", {}).get("type")
        background_types[source_type] = background_types.get(source_type, 0) + 1
        background_files[source_file] = background_files.get(source_file, 0) + 1
        if "lighting_hdri" in background:
            errors.append(f"{metadata_path}: aynı karede ikinci arka plan kaynağı bulundu")
        if source_type == "image":
            if background.get("display_mode") != "camera_facing_plane":
                errors.append(f"{metadata_path}: JPG/PNG kamera arka planına uygulanmamış")
            if floor_type not in {"image_shadow_catcher", "image_background_only"}:
                errors.append(f"{metadata_path}: JPG/PNG karede görünür ikinci zemin bulundu")
            if metadata.get("hdri") is not None:
                errors.append(f"{metadata_path}: JPG/PNG karede ayrıca HDRI kullanılmış")
        elif source_type == "hdri":
            if floor_type not in {"hdri_shadow_catcher", "hdri_background_only"}:
                errors.append(f"{metadata_path}: HDRI karede görünür ikinci zemin bulundu")
            if metadata.get("hdri") != source_file:
                errors.append(f"{metadata_path}: HDRI metadata seçilen kaynakla eşleşmiyor")
            if background.get("applied_name") != source_file:
                errors.append(f"{metadata_path}: Blender world'e bağlı HDRI seçilen dosya değil")
        if "low_resolution" in post:
            low_res_frames += 1
            if post.get("output_resolution") != [640, 640]:
                errors.append(f"{metadata_path}: düşük çözünürlük sonrası tuval değişmiş")
        for environment in metadata.get("environment_objects", []):
            environment_count += 1
            if environment.get("annotated") is not False:
                errors.append(f"{metadata_path}: environment annotated=False değil")
        for cube in metadata.get("cubes", []):
            rotation = cube.get("rotation_euler_rad")
            if not isinstance(rotation, list) or len(rotation) != 3:
                errors.append(f"{metadata_path}: küp X/Y/Z rotation metadata eksik")
            if cube.get("stacked_on") is None:
                half_z = cube.get("half_extents", [None, None, None])[2]
                if half_z is not None and abs((cube.get("pos_z", 0) - half_z) - 0.002) > 0.006:
                    errors.append(f"{metadata_path}: döndürülmüş küp zemine oturmuyor")

    coco_path = os.path.join(output_dir, "coco", "coco_annotations.json")
    coco_images = coco_annotations = 0
    if os.path.isfile(coco_path):
        with open(coco_path, encoding="utf-8") as handle:
            coco = json.load(handle)
        coco_images = len(coco.get("images", []))
        coco_annotations = len(coco.get("annotations", []))
        if any(int(item.get("category_id", 0)) != 1 for item in coco.get("annotations", [])):
            errors.append("COCO içinde küp dışı category_id bulundu")
        category_ids = {int(item.get("id", 0)) for item in coco.get("categories", [])}
        if category_ids != {1}:
            errors.append(f"COCO kategori kümesi {category_ids}, yalnızca {{1}} bekleniyor")
    else:
        errors.append(f"COCO annotation bulunamadı: {coco_path}")

    if metadata_paths and len(background_types) == 1:
        only_type = next(iter(background_types))
        warnings.append(
            f"Yalnızca '{only_type}' arka plan türü kullanılmış; diğer kaynak havuzunun "
            "üretim başlangıcında bulunduğunu kontrol edin"
        )

    summary = {
        "images": len(image_paths),
        "labels": len(label_paths),
        "metadata": len(metadata_paths),
        "low_res_frames": low_res_frames,
        "environment_objects": environment_count,
        "background_types": background_types,
        "background_files": background_files,
        "coco_images": coco_images,
        "coco_annotations": coco_annotations,
        "warnings": warnings,
        "errors": errors,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "output"))
    args = parser.parse_args()
    summary = validate(os.path.abspath(args.output_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
