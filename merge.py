#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paralel worker YOLO çıktılarını birleştirir, train/val/test split yapar
ve dataset istatistik raporu üretir.

Kullanım:
    python merge.py
    python merge.py --output-dir /özel/yol
    python merge.py --no-split          # split yapmadan tek klasörde bırak
"""

import glob
import json
import os
import random
import shutil
import sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

YOLO_SUBDIR         = "yolo"
YOLO_IMAGES_SUBDIR  = "images"
YOLO_LABELS_SUBDIR  = "labels"
YOLO_META_SUBDIR    = "metadata"
COCO_SUBDIR         = "coco"
COCO_IMAGES_SUBDIR  = "images"
COCO_ANNOTATIONS_FILE = "coco_annotations.json"
DATASET_NC          = 1
DATASET_NAMES       = ["cube"]
DATASET_TRAIN_RATIO = 0.75
DATASET_VAL_RATIO   = 0.15
DATASET_TEST_RATIO  = 0.10
SPLIT_SEED          = 42   # Tekrarlanabilir split için

_no_split = "--no-split" in sys.argv

for _i, _a in enumerate(sys.argv):
    if _a == "--output-dir" and _i + 1 < len(sys.argv):
        OUTPUT_DIR = sys.argv[_i + 1]
        break


def merge_yolo_workers(workers_dir, out_img_dir, out_lbl_dir, out_meta_dir):
    """
    workers_dir/worker_*/yolo/{images,labels,metadata}/ klasörlerini tarar,
    dosyaları zero-padded global isimle hedef dizinlere kopyalar.

    Returns:
        list[str]: Birleştirilen görüntülerin stem listesi (örn. "000000")
    """
    worker_dirs = sorted(glob.glob(os.path.join(workers_dir, "worker_*")))
    if not worker_dirs:
        print(f"HATA: {workers_dir} altında worker çıktısı bulunamadı.")
        sys.exit(1)

    print(f"{len(worker_dirs)} worker çıktısı bulundu.")
    os.makedirs(out_img_dir,  exist_ok=True)
    os.makedirs(out_lbl_dir,  exist_ok=True)
    os.makedirs(out_meta_dir, exist_ok=True)

    stems   = []
    counter = 0

    for worker_dir in worker_dirs:
        img_dir  = os.path.join(worker_dir, YOLO_SUBDIR, YOLO_IMAGES_SUBDIR)
        lbl_dir  = os.path.join(worker_dir, YOLO_SUBDIR, YOLO_LABELS_SUBDIR)
        meta_dir = os.path.join(worker_dir, YOLO_SUBDIR, YOLO_META_SUBDIR)

        img_files = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.png"))
        )

        for img_path in img_files:
            src_stem = os.path.splitext(os.path.basename(img_path))[0]
            ext      = os.path.splitext(img_path)[1]
            new_stem = f"{counter:06d}"

            shutil.copy2(img_path, os.path.join(out_img_dir, f"{new_stem}{ext}"))

            lbl_src = os.path.join(lbl_dir,  f"{src_stem}.txt")
            lbl_dst = os.path.join(out_lbl_dir, f"{new_stem}.txt")
            if os.path.isfile(lbl_src):
                shutil.copy2(lbl_src, lbl_dst)
            else:
                open(lbl_dst, "w").close()

            meta_src = os.path.join(meta_dir, f"{src_stem}.json")
            meta_dst = os.path.join(out_meta_dir, f"{new_stem}.json")
            if os.path.isfile(meta_src):
                shutil.copy2(meta_src, meta_dst)

            stems.append(new_stem)
            counter += 1

    print(f"Birleştirme tamamlandı: {counter} görüntü → {out_img_dir}")
    return stems


def merge_coco_workers(workers_dir, output_dir):
    """Worker COCO dosyalarını global image/annotation id'leriyle birleştirir."""
    worker_dirs = sorted(glob.glob(os.path.join(workers_dir, "worker_*")))
    coco_dir = os.path.join(output_dir, COCO_SUBDIR)
    coco_images_dir = os.path.join(coco_dir, COCO_IMAGES_SUBDIR)
    os.makedirs(coco_images_dir, exist_ok=True)
    merged = {
        "info": {"description": "Merged BlenderProc synthetic cube dataset", "version": "2.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "cube", "supercategory": "object"}],
    }
    global_image_id = 1
    global_annotation_id = 1
    global_stem = 0

    for worker_dir in worker_dirs:
        worker_coco = os.path.join(worker_dir, COCO_SUBDIR)
        annotation_path = os.path.join(worker_coco, COCO_ANNOTATIONS_FILE)
        if not os.path.isfile(annotation_path):
            continue
        with open(annotation_path, encoding="utf-8") as handle:
            document = json.load(handle)
        annotations_by_image = {}
        for annotation in document.get("annotations", []):
            if int(annotation.get("category_id", 0)) != 1:
                continue
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

        for image in sorted(document.get("images", []), key=lambda item: item["file_name"]):
            old_image_id = image["id"]
            source_name = os.path.basename(image["file_name"])
            source_path = os.path.join(worker_coco, COCO_IMAGES_SUBDIR, source_name)
            extension = os.path.splitext(source_name)[1] or ".jpg"
            new_name = f"{global_stem:06d}{extension}"
            if os.path.isfile(source_path):
                shutil.copy2(source_path, os.path.join(coco_images_dir, new_name))
            merged["images"].append({
                "id": global_image_id,
                "file_name": f"{COCO_IMAGES_SUBDIR}/{new_name}",
                "width": int(image.get("width", 640)),
                "height": int(image.get("height", 640)),
            })
            for annotation in annotations_by_image.get(old_image_id, []):
                merged_annotation = dict(annotation)
                merged_annotation["id"] = global_annotation_id
                merged_annotation["image_id"] = global_image_id
                merged["annotations"].append(merged_annotation)
                global_annotation_id += 1
            global_image_id += 1
            global_stem += 1

    output_path = os.path.join(coco_dir, COCO_ANNOTATIONS_FILE)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False)
    print(f"COCO birleştirme tamamlandı: {len(merged['images'])} görüntü, "
          f"{len(merged['annotations'])} annotation → {output_path}")
    return merged


def split_dataset(stems, out_img_dir, out_lbl_dir, out_meta_dir, yolo_dir):
    """
    Tüm görüntüleri %75 train / %15 val / %10 test olarak ayırır.
    Her split için ayrı alt dizin oluşturur; dosyalar taşınır (kopyalanmaz).

    Returns:
        dict: {train: [...], val: [...], test: [...]}
    """
    rng = random.Random(SPLIT_SEED)
    shuffled = list(stems)
    rng.shuffle(shuffled)

    n      = len(shuffled)
    n_test = max(1, int(n * DATASET_TEST_RATIO))
    n_val  = max(1, int(n * DATASET_VAL_RATIO))
    n_train = n - n_val - n_test

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    for split_name, split_stems in splits.items():
        s_img  = os.path.join(yolo_dir, YOLO_IMAGES_SUBDIR, split_name)
        s_lbl  = os.path.join(yolo_dir, YOLO_LABELS_SUBDIR, split_name)
        s_meta = os.path.join(yolo_dir, YOLO_META_SUBDIR,   split_name)
        os.makedirs(s_img,  exist_ok=True)
        os.makedirs(s_lbl,  exist_ok=True)
        os.makedirs(s_meta, exist_ok=True)

        for stem in split_stems:
            for src_dir, dst_dir, ext in [
                (out_img_dir,  s_img,  ".jpg"),
                (out_img_dir,  s_img,  ".png"),
                (out_lbl_dir,  s_lbl,  ".txt"),
                (out_meta_dir, s_meta, ".json"),
            ]:
                src = os.path.join(src_dir, f"{stem}{ext}")
                if os.path.isfile(src):
                    shutil.move(src, os.path.join(dst_dir, f"{stem}{ext}"))

    print(f"Split tamamlandı: train={len(splits['train'])}  "
          f"val={len(splits['val'])}  test={len(splits['test'])}")
    return splits


def write_dataset_yaml(yolo_dir, splits):
    """YOLOv8 formatında dataset.yaml üretir."""
    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    abs_yolo  = os.path.abspath(yolo_dir)

    if splits:
        train_path = f"{YOLO_IMAGES_SUBDIR}/train"
        val_path   = f"{YOLO_IMAGES_SUBDIR}/val"
        test_path  = f"{YOLO_IMAGES_SUBDIR}/test"
    else:
        train_path = val_path = test_path = YOLO_IMAGES_SUBDIR

    content = (
        f"# Otomatik üretildi — merge.py\n"
        f"path: {abs_yolo}\n"
        f"train: {train_path}\n"
        f"val:   {val_path}\n"
        f"test:  {test_path}\n"
        f"\n"
        f"nc: {DATASET_NC}\n"
        f"names: {DATASET_NAMES}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"dataset.yaml yazıldı → {yaml_path}")


def _load_labels(lbl_dir):
    """Bir label dizinindeki tüm bbox satırlarını toplar."""
    boxes = []
    for p in glob.glob(os.path.join(lbl_dir, "**/*.txt"), recursive=True):
        with open(p, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([float(v) for v in parts[1:]])
    return boxes


def _load_metadata(meta_dir):
    """Metadata JSON dosyalarından istatistik verilerini toplar."""
    floor_counts = {}
    vis_ratios   = []
    n_skipped    = 0
    n_annotated  = 0

    for p in glob.glob(os.path.join(meta_dir, "**/*.json"), recursive=True):
        try:
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            ft = m.get("floor_type", "unknown")
            floor_counts[ft] = floor_counts.get(ft, 0) + 1
            n_annotated += m.get("n_cubes_annotated", 0)
            n_skipped   += m.get("n_cubes_skipped", 0)
            for c in m.get("cubes", []):
                vr = c.get("visibility_ratio")
                if vr is not None and vr >= 0:
                    vis_ratios.append(vr)
        except Exception:
            pass

    return floor_counts, vis_ratios, n_annotated, n_skipped


def print_stats(total_images, splits, yolo_dir):
    """Dataset istatistik raporu yazdırır."""
    if splits:
        lbl_dirs = [
            os.path.join(yolo_dir, YOLO_LABELS_SUBDIR, s)
            for s in ("train", "val", "test")
        ]
        meta_dirs = [
            os.path.join(yolo_dir, YOLO_META_SUBDIR, s)
            for s in ("train", "val", "test")
        ]
    else:
        lbl_dirs  = [os.path.join(yolo_dir, YOLO_LABELS_SUBDIR)]
        meta_dirs = [os.path.join(yolo_dir, YOLO_META_SUBDIR)]

    all_boxes = []
    for d in lbl_dirs:
        all_boxes.extend(_load_labels(d))

    all_floor, all_vis, total_ann, total_skip = {}, [], 0, 0
    for d in meta_dirs:
        fc, vr, na, ns = _load_metadata(d)
        for k, v in fc.items():
            all_floor[k] = all_floor.get(k, 0) + v
        all_vis.extend(vr)
        total_ann  += na
        total_skip += ns

    n_boxes    = len(all_boxes)
    avg_per_img = n_boxes / max(total_images, 1)

    print("\n" + "=" * 50)
    print("  Dataset İstatistik Raporu")
    print("=" * 50)
    print(f"  Toplam görüntü    : {total_images}")
    if splits:
        print(f"  Train/Val/Test    : {len(splits['train'])} / "
              f"{len(splits['val'])} / {len(splits['test'])}")
    print(f"  Toplam bbox       : {n_boxes}")
    print(f"    Anotate edilen  : {total_ann}")
    print(f"    Atlanan (vis<%40): {total_skip}")
    print(f"  Ort. bbox/görüntü : {avg_per_img:.2f}")

    if all_boxes:
        areas = [(b[2] * 640) * (b[3] * 640) for b in all_boxes]
        small  = sum(1 for a in areas if a <  32 * 32)
        medium = sum(1 for a in areas if  32 * 32 <= a < 128 * 128)
        large  = sum(1 for a in areas if a >= 128 * 128)
        print(f"  Bbox boyut dağılımı:")
        print(f"    Küçük  (<32px)   : {small}  ({100*small//max(n_boxes,1)}%)")
        print(f"    Orta   (32-128px): {medium} ({100*medium//max(n_boxes,1)}%)")
        print(f"    Büyük  (>128px)  : {large}  ({100*large//max(n_boxes,1)}%)")

    if all_vis:
        import statistics
        print(f"  Küp görünürlük    : ort={statistics.mean(all_vis):.3f}  "
              f"min={min(all_vis):.3f}  max={max(all_vis):.3f}")

    if all_floor:
        total_f = sum(all_floor.values())
        floor_str = "  |  ".join(
            f"{k} {v} ({100*v//total_f}%)"
            for k, v in sorted(all_floor.items(), key=lambda x: -x[1])
        )
        print(f"  Zemin dağılımı    : {floor_str}")

    print("=" * 50 + "\n")


def main():
    yolo_dir     = os.path.join(OUTPUT_DIR, YOLO_SUBDIR)
    workers_dir  = os.path.join(OUTPUT_DIR, "workers")
    out_img_dir  = os.path.join(yolo_dir, YOLO_IMAGES_SUBDIR)
    out_lbl_dir  = os.path.join(yolo_dir, YOLO_LABELS_SUBDIR)
    out_meta_dir = os.path.join(yolo_dir, YOLO_META_SUBDIR)

    stems = merge_yolo_workers(workers_dir, out_img_dir, out_lbl_dir, out_meta_dir)
    merge_coco_workers(workers_dir, OUTPUT_DIR)

    splits = None
    if not _no_split and len(stems) > 0:
        splits = split_dataset(stems, out_img_dir, out_lbl_dir, out_meta_dir, yolo_dir)

    write_dataset_yaml(yolo_dir, splits)
    print_stats(len(stems), splits, yolo_dir)


if __name__ == "__main__":
    main()
