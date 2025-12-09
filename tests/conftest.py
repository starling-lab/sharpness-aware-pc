"""
Pytest configuration for hessian_reg tests.
Loads environment variables from .env file.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if it exists
env_file = project_root / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                # Only set if not already set
                if key not in os.environ:
                    os.environ[key] = value
