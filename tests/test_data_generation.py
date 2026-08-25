from battery_prediction.data.generator import generate
import os


def test_generate_seed_reproducible(tmp_path):
    out1 = tmp_path / "d1.csv"
    out2 = tmp_path / "d2.csv"
    generate(n=100, out=str(out1), seed=1)
    generate(n=100, out=str(out2), seed=1)
    with open(out1, "rb") as f1, open(out2, "rb") as f2:
        assert f1.read() == f2.read()
