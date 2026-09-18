import json
import os
import tempfile
import unittest

from PIL import Image

from merge import merge_coco_workers, merge_yolo_workers


class MergeTests(unittest.TestCase):
    def test_worker_yolo_and_coco_merge(self):
        with tempfile.TemporaryDirectory() as temp:
            workers = os.path.join(temp, "workers")
            for worker_id in range(2):
                root = os.path.join(workers, f"worker_{worker_id}")
                for path in (
                    os.path.join(root, "yolo", "images"),
                    os.path.join(root, "yolo", "labels"),
                    os.path.join(root, "yolo", "metadata"),
                    os.path.join(root, "coco", "images"),
                ):
                    os.makedirs(path, exist_ok=True)
                Image.new("RGB", (640, 640), "gray").save(
                    os.path.join(root, "yolo", "images", "000000.jpg")
                )
                Image.new("RGB", (640, 640), "gray").save(
                    os.path.join(root, "coco", "images", "000000.jpg")
                )
                with open(os.path.join(root, "yolo", "labels", "000000.txt"), "w") as handle:
                    handle.write("0 0.5 0.5 0.2 0.2\n")
                with open(os.path.join(root, "yolo", "metadata", "000000.json"), "w") as handle:
                    json.dump({"environment_objects": [{"annotated": False}]}, handle)
                coco = {
                    "images": [{"id": 1, "file_name": "images/000000.jpg", "width": 640, "height": 640}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500},
                        {"id": 2, "image_id": 1, "category_id": 0, "bbox": [1, 1, 5, 5], "area": 25},
                    ],
                }
                with open(os.path.join(root, "coco", "coco_annotations.json"), "w") as handle:
                    json.dump(coco, handle)

            yolo = os.path.join(temp, "yolo")
            stems = merge_yolo_workers(
                workers, os.path.join(yolo, "images"), os.path.join(yolo, "labels"),
                os.path.join(yolo, "metadata"),
            )
            merged = merge_coco_workers(workers, temp)
            self.assertEqual(stems, ["000000", "000001"])
            self.assertEqual(len(merged["images"]), 2)
            self.assertEqual(len(merged["annotations"]), 2)
            self.assertEqual({item["category_id"] for item in merged["annotations"]}, {1})
            self.assertEqual([item["id"] for item in merged["images"]], [1, 2])


if __name__ == "__main__":
    unittest.main()

