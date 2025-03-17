from jax.lib import xla_bridge

platform = xla_bridge.get_backend().platform
if platform != "GPU":
    print(f"GPU support not available, using {platform}")
else:
    print("GPU support available!")
