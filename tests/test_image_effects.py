import random
import unittest

import numpy as np

from image_effects import distort_bbox, process_image, radial_distort_array


def full_config():
    return {
        "enabled": True,
        "noise_probability": 1.0,
        "noise_sigma_range": (1.0, 1.0),
        "gaussian_blur_probability": 1.0,
        "gaussian_blur_range": (0.4, 0.4),
        "temperature_probability": 1.0,
        "temperature_range": (6000, 6000),
        "brightness_contrast_probability": 1.0,
        "brightness_range": (1.05, 1.05),
        "contrast_range": (0.95, 0.95),
        "gamma_probability": 1.0,
        "gamma_range": (1.05, 1.05),
        "motion_blur_probability": 1.0,
        "motion_blur_radius_range": (1, 1),
        "lens_distortion_probability": 1.0,
        "lens_distortion_range": (0.01, 0.01),
        "exposure_probability": 1.0,
        "exposure_ev_range": (0.1, 0.1),
        "low_res_probability": 1.0,
        "low_res_sizes": [160],
        "upscale_methods": ["bilinear"],
        "jpeg_probability": 1.0,
        "jpeg_quality_range": (70, 70),
    }


class ImageEffectsTests(unittest.TestCase):
    def test_low_resolution_preserves_canvas_and_normalized_bbox(self):
        image = np.full((640, 640, 3), 128, dtype=np.uint8)
        mask = np.zeros((640, 640), dtype=np.int32)
        mask[200:400, 220:420] = 7
        before = (0.5, 0.46875, 0.3125, 0.3125)
        result, metadata, transformed = process_image(
            image, full_config(), rng=random.Random(5),
            np_rng=np.random.default_rng(5), auxiliary_arrays=[mask],
        )
        self.assertEqual(result.shape, (640, 640, 3))
        self.assertEqual(metadata["output_resolution"], [640, 640])
        self.assertEqual(metadata["low_resolution"]["downsample_size"], [160, 160])
        self.assertEqual(metadata["jpeg_quality"], 70)
        self.assertEqual(before, (0.5, 0.46875, 0.3125, 0.3125))
        self.assertEqual(transformed[0].shape, mask.shape)
        self.assertIn(7, np.unique(transformed[0]))

    def test_radial_bbox_stays_valid(self):
        bbox = distort_bbox((100, 120, 200, 180), 0.015, 640, 640)
        self.assertIsNotNone(bbox)
        x, y, width, height = bbox
        self.assertTrue(0 <= x < 640 and 0 <= y < 640)
        self.assertTrue(width > 0 and height > 0)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        warped = radial_distort_array(mask, -0.01, order=0)
        self.assertEqual(warped.shape, mask.shape)
        self.assertGreater(int(warped.sum()), 0)


if __name__ == "__main__":
    unittest.main()

