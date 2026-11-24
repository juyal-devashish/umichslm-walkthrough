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

# Token-Superposition Training (TST) hook: when the model config has
# tst_enabled=True, auto-attach TSTTrainerCallback to any Trainer instance
# created by LLaMA Factory. The recovery cutover happens within the run.
from transformers import Trainer as _HFTrainer
from umich_slm.callbacks import TSTTrainerCallback as _TSTCB

_orig_trainer_init = _HFTrainer.__init__

def _patched_trainer_init(self, *args, **kwargs):
    _orig_trainer_init(self, *args, **kwargs)
    cfg = getattr(getattr(self, 'model', None), 'config', None)
    if cfg is not None and bool(getattr(cfg, 'tst_enabled', False)):
        bag = int(getattr(cfg, 'tst_bag_size', 1))
        ratio = float(getattr(cfg, 'tst_recovery_ratio', 0.3))
        if bag >= 2 and 0.0 < ratio < 1.0:
            cb = _TSTCB(enabled=True, bag_size=bag, recovery_ratio=ratio)
            self.add_callback(cb)
            print(f"[umich_slm] attached TSTTrainerCallback (s={{bag}}, r={{ratio}})")
    # Delphi recipe: swap optimizer + scheduler at first training step.
    if cfg is not None and bool(getattr(cfg, 'delphi_recipe', False)):
        try:
            from umich_slm.recipe.delphi import DelphiRecipeConfig
            from umich_slm.recipe.integrations import hf_trainer_optim_factories
        except ImportError as _e:
            print(f"[umich_slm] delphi_recipe requested but imports failed: {{_e}}")
        else:
            d_tokens = float(getattr(cfg, 'delphi_run_tokens', 0.0))
            d_batch = int(getattr(cfg, 'delphi_run_batch', 0))
            d_steps = int(getattr(cfg, 'delphi_run_total_steps', 0))
            if d_tokens > 0 and d_batch > 0 and d_steps > 0:
                rc = DelphiRecipeConfig(
                    eta0_proj=float(getattr(cfg, 'delphi_eta0_proj', 6.30e-3)),
                    eta0_adam=float(getattr(cfg, 'delphi_eta0_adam', 6.56e-4)),
                    batch_ref=int(getattr(cfg, 'delphi_batch_ref', 64)),
                    token_ref=float(getattr(cfg, 'delphi_token_ref', 2.5e9)),
                    horizon_exp=float(getattr(cfg, 'delphi_horizon_exp', 0.30)),
                    epsilon0=float(getattr(cfg, 'delphi_epsilon0', 1.85e-8)),
                    beta2_max=float(getattr(cfg, 'delphi_beta2_max', 0.9999)),
                    beta2_min=float(getattr(cfg, 'delphi_beta2_min', 0.9)),
                    grad_clip=float(getattr(cfg, 'delphi_grad_clip', 0.1)),
                    warmup_frac=float(getattr(cfg, 'delphi_warmup_frac', 0.10)),
                    decay_frac=float(getattr(cfg, 'delphi_decay_frac', 0.20)),
                    schedule=str(getattr(cfg, 'delphi_schedule', 'wsd')),
                )
                opt_name = str(getattr(cfg, 'delphi_optimizer', 'adamh'))
                opt, sched = hf_trainer_optim_factories(
                    self.model, tokens=d_tokens, batch=d_batch,
                    total_steps=d_steps, config=rc, optimizer_name=opt_name,
                )
                self.optimizer = opt
                self.lr_scheduler = sched
                print(f"[umich_slm] Delphi recipe active (opt={{opt_name}}, sched={{rc.schedule}}, "
                      f"T={{d_tokens:.2e}}, B={{d_batch}}, steps={{d_steps}})")

_HFTrainer.__init__ = _patched_trainer_init

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

