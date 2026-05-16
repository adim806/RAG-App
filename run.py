"""
DevMind entry point.

Run from the devmind/ directory:
    python run.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from web.app import app  # noqa: E402

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
