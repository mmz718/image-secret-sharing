"""验证门限、独立实现、域边界及文件完整性。"""
import itertools
import json
from pathlib import Path
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import sharing as ss
from experiment import make_preview


class SharingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.array([[[0, 255, 1, 0], [255, 0, 254, 255]],
                               [[20, 100, 200, 128], [30, 150, 80, 1]]], dtype=np.uint8)

    def test_all_authorized_combinations(self):
        for k, n in ((3, 5), (4, 6)):
            for method in ("shamir", "brickell"):
                shares = ss.split(self.image, method, k, n)
                for count in range(k, n + 1):
                    for ids in itertools.combinations(range(1, n + 1), count):
                        with self.subTest(k=k, method=method, ids=ids):
                            restored = ss.recover(shares[np.array(ids) - 1], ids, method, k, n)
                            np.testing.assert_array_equal(restored, self.image)

    def test_all_unauthorized_combinations(self):
        for k, n in ((3, 5), (4, 6)):
            target = np.eye(1, k, dtype=np.int64)[0]
            public = ss.vectors(k, n)
            for count in range(1, k):
                for ids in itertools.combinations(range(1, n + 1), count):
                    with self.assertRaises(ValueError):
                        ss.solve_mod(public[np.array(ids) - 1].T, target)

    def test_two_secrets_match_same_shares(self):
        public = ss.vectors(3, 5)
        known = np.array([127, 168])
        matrix = np.vstack((public[:2], [1, 0, 0]))
        for secret in (100, 101):
            coeff = ss.solve_mod(matrix, np.append(known, secret))
            np.testing.assert_array_equal(public[:2] @ coeff % ss.P, known)
            self.assertEqual(coeff[0], secret)

    def test_equivalent_under_same_randomness(self):
        fixed = np.arange(2 * self.image.size, dtype=np.int64).reshape(2, -1) % ss.P
        with patch("sharing.random_field", return_value=fixed):
            a = ss.split(self.image, "shamir", 3, 5)
            b = ss.split(self.image, "brickell", 3, 5)
        np.testing.assert_array_equal(a, b)

    def test_scalar_example(self):
        image = np.array([[100]], dtype=np.uint8)
        with patch("sharing.random_field", return_value=np.array([[20], [7]], dtype=np.int64)):
            for method in ("shamir", "brickell"):
                shares = ss.split(image, method, 3, 5)
                self.assertEqual(shares.reshape(-1).tolist(), [127, 168, 223, 35, 118])

    def test_random_sampler_rejects_biased_tail(self):
        with patch("sharing.secrets.token_bytes", side_effect=[np.array([65535, 256], dtype="<u2").tobytes(),
                                                               np.array([257], dtype="<u2").tobytes()]):
            self.assertEqual(ss.random_field((2,)).tolist(), [256, 0])

    def test_zero_highest_coefficient_allowed(self):
        with patch("sharing.random_field", return_value=np.zeros((2, self.image.size), dtype=np.int64)):
            for method in ("shamir", "brickell"):
                shares = ss.split(self.image, method, 3, 5)
                np.testing.assert_array_equal(ss.recover(shares[:3], [1, 2, 3], method, 3, 5), self.image)

    def test_invalid_parameters(self):
        for k, n in ((1, 5), (4, 3), (2, 257), (True, 5)):
            with self.assertRaises(ValueError):
                ss.check(k, n)

    def test_invalid_image(self):
        for image in (np.array([], dtype=np.uint8), np.array([[1.5]]), np.zeros((1, 1, 1, 1), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                ss.split(image, "shamir", 2, 3)

    def test_duplicate_and_insufficient_ids(self):
        shares = ss.split(self.image, "shamir", 3, 5)
        for ids in ([], [1, 1, 2], [1, 2], [0, 2, 3], [1, 2, 6]):
            with self.assertRaises(ValueError):
                ss.recover(shares[:len(ids)], ids, "shamir", 3, 5)

    def test_field_boundary_and_disk_roundtrip(self):
        image = np.array([[255]], dtype=np.uint8)
        with patch("sharing.random_field", return_value=np.ones((1, 1), dtype=np.int64)):
            shares = ss.split(image, "shamir", 2, 3)
        self.assertEqual(shares[0, 0, 0], 256)
        with tempfile.TemporaryDirectory() as folder:
            paths = ss.save_shares(folder, shares, "shamir", 2, 3, "L")
            saved, ids, meta = ss.load_shares([paths[2], paths[0]])
            self.assertEqual(saved[1, 0, 0], 256)
            np.testing.assert_array_equal(ss.recover(saved, ids, meta["method"], 2, 3), image)

    def test_mixed_sessions_rejected(self):
        shares = ss.split(self.image, "shamir", 3, 5)
        with tempfile.TemporaryDirectory() as folder:
            a = ss.save_shares(Path(folder) / "a", shares, "shamir", 3, 5, "RGBA")
            b = ss.save_shares(Path(folder) / "b", shares, "shamir", 3, 5, "RGBA")
            with self.assertRaises(ValueError):
                ss.load_shares([a[0], b[1]])

    def test_corrupt_share_rejected(self):
        shares = ss.split(self.image, "shamir", 3, 5)[:3]
        shares[0, 0, 0, 0] = 257
        with self.assertRaises(ValueError):
            ss.recover(shares, [1, 2, 3], "shamir", 3, 5)

    def test_precomputation_binding(self):
        shares = ss.split(self.image, "brickell", 3, 5)
        prepared = ("brickell", (1, 2, 3), ss.weights("brickell", 3, 5, [1, 2, 3]))
        np.testing.assert_array_equal(ss.recover(shares[:3], [1, 2, 3], "brickell", 3, 5, prepared), self.image)
        with self.assertRaises(ValueError):
            ss.recover(shares[1:4], [2, 3, 4], "brickell", 3, 5, prepared)

    def test_domain_last_coordinate(self):
        for method in ("shamir", "brickell"):
            shares = ss.split(self.image, method, 2, 256)
            np.testing.assert_array_equal(ss.recover(shares[[255, 0]], [256, 1], method, 2, 256), self.image)

    def test_previews_support_all_modes(self):
        for mode, image in (("L", self.image[..., 0]), ("RGB", self.image[..., :3]), ("RGBA", self.image)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "preview.png"
                make_preview(image, ss.split(image, "shamir", 2, 3), image, mode, path)
                with Image.open(path) as preview:
                    self.assertEqual(preview.mode, "RGB")

    def test_numpy_buffer_is_traced(self):
        tracemalloc.start()
        try:
            array = np.empty(1000000, dtype=np.uint16)
            self.assertGreaterEqual(tracemalloc.get_traced_memory()[1], array.nbytes)
        finally:
            tracemalloc.stop()


if __name__ == "__main__":
    out = Path("results")
    out.mkdir(exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SharingTests)
    with (out / "tests.log").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    summary = {"methods": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "successful": result.wasSuccessful()}
    (out / "tests.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    if not result.wasSuccessful():
        print((out / "tests.log").read_text(encoding="utf-8"))
        raise SystemExit(1)
