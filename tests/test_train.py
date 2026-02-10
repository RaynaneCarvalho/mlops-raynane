import os
import subprocess
import sys


def test_train_script_runs():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["LANG"] = "en_US.UTF-8"

    result = subprocess.run(
        [sys.executable, "train.py", "--max_depth", "2"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert os.path.exists("models/diamond_price_model.joblib")
