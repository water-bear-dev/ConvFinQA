import importlib.util
import json
import os
import sys
import time


def log(message: str, data: dict) -> None:
    payload = {
        "sessionId": "6857b2",
        "runId": "import-debug",
        "hypothesisId": "H1-H5",
        "location": "debug_import.py:log",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(
        "/Users/andrewpham/Documents/GitHub/water-bear-dev/.cursor/debug-6857b2.log",
        "a",
    ) as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    log(
        "python_env",
        {
            "sys_executable": sys.executable,
            "cwd": os.getcwd(),
            "sys_path_head": sys.path[:5],
            "src_spec": str(importlib.util.find_spec("src")),
            "src_init_exists": os.path.exists(
                "/Users/andrewpham/Documents/GitHub/water-bear-dev/src/__init__.py"
            ),
        },
    )


if __name__ == "__main__":
    main()

