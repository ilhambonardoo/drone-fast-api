"""Pemeriksaan statis agar fungsi numerik pada kedua adapter tetap sama.
Tidak mengimpor TensorFlow, mengakses Supabase, atau menjalankan server.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = (
    "prepare_mapir_patch",
    "extract_mapir_grn",
    "mapir_jpg_to_model_scale",
    "calculate_ndvi_and_spectral",
    "preprocess_for_cnn",
    "create_ndvi_heatmap",
)


def read_functions(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class PipelineConsistencyTests(unittest.TestCase):
    def test_six_numeric_functions_match(self):
        local = read_functions("main.py")
        cloud = read_functions("main_supabase.py")
        for name in FUNCTIONS:
            with self.subTest(function=name):
                self.assertIn(name, local)
                self.assertIn(name, cloud)
                self.assertEqual(local[name], cloud[name])

    def test_both_entrypoints_compile(self):
        for filename in ("main.py", "main_supabase.py"):
            with self.subTest(filename=filename):
                path = ROOT / filename
                compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
