import os

# Tests run on CPU everywhere; training on a GPU is done outside pytest.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
