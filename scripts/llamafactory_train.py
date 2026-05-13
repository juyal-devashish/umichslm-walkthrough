#!/usr/bin/env python3
"""Wrapper script for LlamaFactory that ensures UMichSLM model is registered.

This script registers the UMichSLM model with Transformers before calling
LlamaFactory, ensuring that custom model architectures are recognized.

Usage:
    # Instead of: llamafactory-cli train config.yaml
    # Use: python scripts/llamafactory_train.py train config.yaml

    # All other llamafactory-cli commands work the same way
    python scripts/llamafactory_train.py train configs/llama_factory/umichslm_pretrain.yaml
"""

import sys
import subprocess
from pathlib import Path

# Add the project root to Python path to ensure umich_slm can be imported
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Register the model in this process
try:
    # Import the package - this triggers registration in __init__.py
    import umich_slm  # noqa: F401
    print("✓ UMichSLM model registered with Transformers")
except ImportError as e:
    print(f"✗ Error: Could not import umich_slm: {e}")
    print("  Make sure the package is installed: pip install -e .")
    sys.exit(1)

# Set PYTHONPATH to ensure the package can be imported in the subprocess
# This is needed because the saved model files import from umich_slm
import os
env = os.environ.copy()
pythonpath = env.get("PYTHONPATH", "")
if pythonpath:
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{pythonpath}"
else:
    env["PYTHONPATH"] = str(project_root)

# Create a Python script that imports the package before running LlamaFactory
# This ensures registration happens in the subprocess
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(f"""#!/usr/bin/env python3
import sys
sys.path.insert(0, r'{project_root}')
import umich_slm  # Register the model
# Now run LlamaFactory CLI
from llamafactory.cli import main
if __name__ == '__main__':
    sys.exit(main())
""")
    temp_script = f.name

try:
    import stat
    os.chmod(temp_script, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    cmd = [sys.executable, temp_script] + sys.argv[1:]
    result = subprocess.run(cmd, env=env)
    os.unlink(temp_script)  # Clean up
    sys.exit(result.returncode)
except FileNotFoundError:
    print("✗ Error: llamafactory not found. Install it with: pip install llamafactory")
    if os.path.exists(temp_script):
        os.unlink(temp_script)
    sys.exit(1)
except Exception as e:
    print(f"✗ Error running LlamaFactory: {e}")
    if os.path.exists(temp_script):
        os.unlink(temp_script)
    sys.exit(1)

