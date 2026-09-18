import ast
from collections import deque
import os
from pathlib import Path
import random
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "generate.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        selector_node = next(
            node for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "select_background_source"
        )
        namespace = {
            "os": os,
            "random": random,
            "_recent_background_paths": deque(maxlen=6),
        }
        exec(compile(ast.Module(body=[selector_node], type_ignores=[]), "generate.py", "exec"), namespace)
        cls.selector = staticmethod(namespace["select_background_source"])
        cls.selector_globals = namespace
        scanner_node = next(
            node for node in cls.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "scan_files"
        )
        scanner_namespace = {"os": os}
        exec(
            compile(ast.Module(body=[scanner_node], type_ignores=[]), "generate.py", "exec"),
            scanner_namespace,
        )
        cls.scan_files = staticmethod(scanner_namespace["scan_files"])

    def test_required_cli_and_config_contracts_exist(self):
        for token in (
            "SOLID_COLOR_PROBABILITY = 0.30",
            "LOW_RES_PROBABILITY",
            "LOW_RES_SIZES",
            "JPEG_COMPRESSION_PROBABILITY",
            "JPEG_QUALITY_RANGE",
            "BACKGROUND_DIVERSITY_WINDOW = 6",
            "BACKGROUND_ASSET_DIRS = [HDRI_DIR, IMAGE_BACKGROUND_DIR]",
            '"--quality-profile"',
            '"--low-res-probability"',
            '"--jpeg-quality-range"',
        ):
            self.assertIn(token, self.source)

    def test_environment_is_background_and_cubes_are_foreground(self):
        self.assertIn('obj.set_cp("category_id", 0)', self.source)
        self.assertIn('cube.set_cp("category_id", 1)', self.source)
        self.assertIn('"annotated": False', self.source)

    def test_orientation_distribution(self):
        self.assertIn("CUBE_ORIENTATION_WEIGHTS = [0.40, 0.40, 0.20]", self.source)
        self.assertIn('"rotation_euler_rad"', self.source)
        self.assertIn('"rotation_euler_deg"', self.source)

    def test_each_frame_uses_exactly_one_background_source(self):
        self.assertIn(
            "select_background_source(\n                hdri_files, image_background_files",
            self.source,
        )
        self.assertIn("_build_image_backdrop_material", self.source)
        self.assertIn("create_image_backdrop", self.source)
        self.assertIn("background_meta = setup_background(background_source)", self.source)
        self.assertIn('if source_type == "hdri"', self.source)
        self.assertIn('if source_type == "image"', self.source)
        self.assertNotIn("lighting_hdri", self.source)
        self.assertNotIn("BACKGROUND_HDRI_WEIGHT", self.source)
        self.assertNotIn("BACKGROUND_IMAGE_WEIGHT", self.source)
        self.assertNotIn("random.choices(kinds", self.source)

    def test_hdri_world_is_rebuilt_for_the_selected_file(self):
        self.assertIn('environment.image = bpy.data.images.load(hdri_path', self.source)
        self.assertIn('"applied_name": os.path.basename(applied_path)', self.source)
        function = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_apply_world_hdri"
        )
        function_source = ast.get_source_segment(self.source, function)
        self.assertIn("nodes.clear()", function_source)
        self.assertNotIn("set_world_background_hdr_img", function_source)

    def test_background_selection_is_seed_reproducible(self):
        hdri_files = ["one.exr", "two.exr", "three.exr"]
        image_files = ["one.png", "two.jpg", "three.jpeg"]

        def sequence():
            self.selector_globals["_recent_background_paths"].clear()
            random.seed(42)
            return [self.selector(hdri_files, image_files) for _ in range(8)]

        first = sequence()
        second = sequence()
        self.assertEqual(first, second)
        self.assertTrue(all(item["source_type"] in {"hdri", "image"} for item in first))
        self.assertTrue(all(item["path"] and item["file_name"] for item in first))

    def test_each_file_has_equal_chance_in_the_combined_pool(self):
        hdri_files = [f"scene_{index}.exr" for index in range(9)]
        image_files = ["single.png"]
        self.selector_globals["_recent_background_paths"].clear()
        random.seed(123)
        selected = [self.selector(hdri_files, image_files) for _ in range(1000)]
        image_count = sum(item["source_type"] == "image" for item in selected)
        self.assertGreater(image_count, 60)
        self.assertLess(image_count, 140)

    def test_background_scan_is_recursive_and_case_insensitive(self):
        with tempfile.TemporaryDirectory() as root:
            nested = Path(root) / "nested"
            nested.mkdir()
            (nested / "one.PNG").touch()
            (nested / "two.jpeg").touch()
            (nested / "ignored.txt").touch()
            found = self.scan_files(root, ["*.png", "*.jpg", "*.jpeg"])
        self.assertEqual([Path(path).name for path in found], ["one.PNG", "two.jpeg"])

    def test_blenderproc_can_resolve_local_helper_modules(self):
        self.assertIn("sys.path.insert(0, BASE_DIR)", self.source)
        self.assertNotIn("background_sources", self.source)


if __name__ == "__main__":
    unittest.main()
