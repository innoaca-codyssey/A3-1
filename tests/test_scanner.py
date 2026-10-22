"""고정 도형과 실제 입출력으로 스캐너를 검증한다."""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

import scanner


class ScannerTests(unittest.TestCase):
    def make_doc_image(self, w=600, h=450):
        img = np.full((h, w, 3), 40, np.uint8)
        pts = np.array([[120, 70], [490, 110], [450, 360], [90, 320]], np.int32)
        cv2.fillConvexPoly(img, pts, (230, 230, 230))
        cv2.polylines(img, [pts], True, (255, 255, 255), 4)
        for y in range(120, 320, 35):
            cv2.line(img, (150, y), (420, y + 10), (160, 160, 160), 2)
        return img, pts.astype(np.float32)

    def test_order_points_axis_aligned(self):
        pts = np.array([[10, 10], [90, 10], [90, 80], [10, 80]], np.float32)
        shuffled = pts[[2, 0, 3, 1]]
        out = scanner.order_points(shuffled)
        self.assertTrue(np.allclose(out, pts))

    def test_order_points_diamond(self):
        pts = np.array([[50, 10], [90, 50], [50, 90], [10, 50]], np.float32)
        out = scanner.order_points(pts[[1, 3, 0, 2]])
        expected = np.array([[10, 50], [50, 10], [90, 50], [50, 90]], np.float32)
        self.assertTrue(np.allclose(out, expected))

    def test_order_points_rotated_convex(self):
        pts = np.array([[30, 20], [100, 40], [80, 120], [10, 90]], np.float32)
        out = scanner.order_points(pts[[2, 0, 3, 1]])
        self.assertEqual(out.shape, (4, 2))
        self.assertLessEqual(out[0, 1], out[3, 1] + 1e-5)

    def test_order_points_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            scanner.order_points(np.array([[0, 0], [1, 1], [1, 1], [0, 1]], np.float32))

    def test_order_points_degenerate_rejected(self):
        with self.assertRaises(ValueError):
            scanner.order_points(np.array([[0, 0], [1, 0], [2, 0], [3, 0]], np.float32))

    def test_numpy_blur_loop_vectorized_equal_gray(self):
        img = np.arange(64, dtype=np.float64).reshape(8, 8)
        a = scanner.numpy_gaussian_blur(img, 5, 1.1, vectorized=False)
        b = scanner.numpy_gaussian_blur(img, 5, 1.1, vectorized=True)
        self.assertTrue(np.allclose(a, b, atol=1e-10))

    def test_numpy_blur_loop_vectorized_equal_rgb(self):
        img = np.dstack([np.arange(36).reshape(6, 6)] * 3).astype(np.float64)
        a = scanner.numpy_gaussian_blur(img, 3, 0.8, vectorized=False)
        b = scanner.numpy_gaussian_blur(img, 3, 0.8, vectorized=True)
        self.assertTrue(np.allclose(a, b, atol=1e-10))

    def test_numpy_blur_matches_opencv_same_kernel_and_boundary(self):
        img = np.random.RandomState(0).rand(20, 22).astype(np.float64)
        out = scanner.numpy_gaussian_blur(img, 5, 1.0, vectorized=True)
        k = cv2.getGaussianKernel(5, 1.0)
        ker = k @ k.T
        ref = cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REFLECT_101)
        self.assertTrue(np.allclose(out, ref, atol=1e-8))

    def test_speed_test_has_measured_values(self):
        img = np.zeros((20, 20), np.float64)
        r = scanner.gaussian_blur_speed_test(img, 3, 1.0, repeats=1)
        self.assertIn("loop", r)
        self.assertGreaterEqual(r["loop"]["seconds"], 0.0)

    def test_find_document_success(self):
        img, _ = self.make_doc_image()
        stages, corners = scanner.find_document(img, min_area_ratio=0.05)
        self.assertIn("edges", stages)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (4, 2))

    def test_find_document_blank_failure(self):
        img = np.zeros((300, 400, 3), np.uint8)
        stages, corners = scanner.find_document(img, min_area_ratio=0.05)
        self.assertIn("contours", stages)
        self.assertIsNone(corners)

    def test_warp_document_outputs(self):
        img, pts = self.make_doc_image()
        warped, binary = scanner.warp_document(img, pts)
        self.assertGreater(warped.shape[0], 50)
        self.assertEqual(binary.ndim, 2)
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))

    def test_scan_image_saves_outputs(self):
        img, _ = self.make_doc_image()
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "doc.png")
            cv2.imwrite(src, img)
            res = scanner.scan_image(
                src, os.path.join(td, "out"), {"min_area_ratio": 0.05}
            )
            self.assertTrue(res["detected"])
            self.assertTrue(os.path.exists(res["files"]["binary"]))
            self.assertTrue(os.path.exists(res["files"]["warped"]))

    def test_scan_image_not_found(self):
        with tempfile.TemporaryDirectory() as td:
            res = scanner.scan_image(os.path.join(td, "missing.png"), td, {})
            self.assertFalse(res["detected"])
            self.assertEqual(res["status"], "not_found")

    def test_export_pdf_magic(self):
        img, _ = self.make_doc_image()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "a.png")
            cv2.imwrite(p, img)
            pdf = os.path.join(td, "out.pdf")
            scanner.export_pdf([p], pdf)
            with open(pdf, "rb") as f:
                magic = f.read(5)
            self.assertEqual(magic, b"%PDF-")

    def test_main_headless_success(self):
        img, _ = self.make_doc_image()
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "doc.png")
            cv2.imwrite(src, img)
            code = scanner.main(
                [
                    "--image",
                    src,
                    "--headless",
                    "--output",
                    os.path.join(td, "out"),
                    "--min-area-ratio",
                    "0.05",
                ]
            )
            self.assertEqual(code, 0)

    def test_main_bad_args(self):
        code = scanner.main(["--headless"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
