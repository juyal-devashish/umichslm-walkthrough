#!/usr/bin/env python3
"""Register UMichSLM model with Transformers before LlamaFactory loads it.

This script ensures that the umich_slm package is imported and registered
with Transformers' Auto classes before LlamaFactory tries to load the model.

Usage:
    # Import this module before running LlamaFactory
    python -c "import sys; sys.path.insert(0, 'scripts'); from register_model import register_umichslm; register_umichslm()"

    # Or use as a module
    python scripts/register_model.py
"""


def register_umichslm():
    """Register UMichSLM model classes with Transformers Auto classes.

    This function imports the umich_slm package, which triggers the
    registration of model classes with Transformers' AutoConfig, AutoModel,
    and AutoModelForCausalLM.

    This should be called before LlamaFactory tries to load the model.
    """
    try:
        # Import the package - this triggers registration in __init__.py
        import umich_slm  # noqa: F401

        # Verify registration worked
        from transformers import AutoConfig
        if "umichslm" in AutoConfig._model_mapping:
            print("✓ UMichSLM model successfully registered with Transformers")
            return True
        else:
            print("⚠ Warning: UMichSLM model type not found in AutoConfig mapping")
            return False
    except ImportError as e:
        print(f"✗ Error importing umich_slm: {e}")
        print("  Make sure the package is installed: pip install -e .")
        return False
    except Exception as e:
        print(f"✗ Error registering model: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = register_umichslm()
    sys.exit(0 if success else 1)

