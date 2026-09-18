import unittest

import numpy as np

from texture import MATERIAL_PROFILES, SURFACE_PATTERNS, generate_variant


class TexturePatternTests(unittest.TestCase):
    def test_every_pattern_generates_albedo_normal_and_metadata(self):
        profile = MATERIAL_PROFILES["PLA_MATTE"]
        for index, pattern in enumerate(SURFACE_PATTERNS):
            with self.subTest(pattern=pattern):
                albedo, normal, metadata = generate_variant(
                    "PLA_MATTE", "red", (195, 42, 38), profile,
                    seed=100 + index, size=48, pattern_type=pattern,
                    secondary_rgb=(35, 90, 210),
                )
                self.assertEqual(albedo.size, (48, 48))
                self.assertEqual(normal.size, (48, 48))
                self.assertEqual(metadata["pattern_type"], pattern)
                self.assertEqual(metadata["secondary_rgb"], [35, 90, 210])
                self.assertGreater(np.asarray(albedo).std(), 0.0)


if __name__ == "__main__":
    unittest.main()

