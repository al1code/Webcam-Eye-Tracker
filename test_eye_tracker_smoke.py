import unittest

import eye_tracker


class EyeTrackerSmokeTests(unittest.TestCase):
    def test_build_heatmap_normalizes_values(self):
        heatmap = eye_tracker.build_heatmap(
            [(10, 10, 0.0), (10, 10, 0.1), (20, 20, 0.2)],
            40,
            40,
            sigma=1,
        )
        self.assertEqual(heatmap.shape, (40, 40))
        self.assertGreaterEqual(float(heatmap.max()), 0.99)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)

    def test_zone_analysis_returns_percentages(self):
        zones = eye_tracker.zone_analysis(
            [(10, 10, 0.0), (90, 10, 0.1), (50, 90, 0.2)],
            100,
            100,
        )
        self.assertAlmostEqual(float(zones.sum()), 100.0, places=4)

    def test_fixation_analysis_keeps_last_cluster(self):
        points = [
            (100, 100, 0.00),
            (102, 101, 0.10),
            (101, 99, 0.25),
        ]
        fixations = eye_tracker.fixation_analysis(points, radius=10, min_dur=0.15)
        self.assertEqual(len(fixations), 1)
        self.assertAlmostEqual(fixations[0][2], 0.25, places=2)

    def test_fit_calibration_model_maps_points_close_to_targets(self):
        targets = [
            (0, 0),
            (960, 0),
            (1920, 0),
            (0, 540),
            (960, 540),
            (1920, 540),
            (0, 1080),
            (960, 1080),
            (1920, 1080),
        ]
        samples = {}
        for idx, (tx, ty) in enumerate(targets):
            rx = 0.35 + (tx / 1920.0) * 0.30
            ry = 0.35 + (ty / 1080.0) * 0.24
            samples[idx] = [(rx, ry)] * 12

        fit = eye_tracker.fit_calibration_model(samples, targets)
        self.assertIsNotNone(fit)
        coeffs_x, coeffs_y, fit_error = fit
        self.assertLess(fit_error, 1.0)

        x, y = eye_tracker.map_ratio_to_screen(
            0.35 + (960 / 1920.0) * 0.30,
            0.35 + (540 / 1080.0) * 0.24,
            1920,
            1080,
            coeffs_x,
            coeffs_y,
            0.4,
            0.6,
            0.4,
            0.6,
        )
        self.assertAlmostEqual(x, 960, delta=3)
        self.assertAlmostEqual(y, 540, delta=3)


if __name__ == "__main__":
    unittest.main()
