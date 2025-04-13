from jax.extend.backend import get_backend

platform = get_backend().platform
if platform != "GPU":
    print(f"GPU support not available, using {platform}")
else:
    print("GPU support available!")
