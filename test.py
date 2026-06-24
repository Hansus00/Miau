import jax
import jax.numpy as jnp
from microjax.inverse_ray.lightcurve import mag_binary

print(jax.devices())
print(jax.default_backend())

t = jnp.linspace(-1.0, 1.0, 100)
w = t + 0.1j
mu = mag_binary(w, 1e-3, s=1.0, q=1e-3)

print(mu[:5])
print(mu.device)