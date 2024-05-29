import pytest
import jax.numpy as jnp
import optimistix as optx

from jaxtyping import Array
from schedule_free_optx import ScheduleFreeAdamW, ScheduleFreeSGD

# This is a smokescreen test to make sure the stepsize-free AdamW and SGD
# algorithms work at all. It is not inteded to actually test their runtimes


def coupled_rosenbrock(y: Array, args):
    del args
    diff_y = 100 * jnp.sum((y[1:] - y[:-1]) ** 2)
    diff_1 = jnp.sum((y - 1) ** 2)
    return diff_y + diff_1


def test_runs(learning_rate: float = 1e-4, rtol: float = 1e-8, atol: float = 1e-9):
    y0 = jnp.zeros(100)
    gd_out = optx.minimise(
        coupled_rosenbrock,
        optx.GradientDescent(learning_rate, rtol, atol),
        y0,
        max_steps=None,
    )
    schedule_free_adam_out = optx.minimise(
        coupled_rosenbrock,
        ScheduleFreeAdamW(
            learning_rate,
            beta_1=0.99,
            beta_2=0.9,
            warmup_steps=1000,
            weight_decay=0.01,
            rtol=rtol,
            atol=atol,
        ),
        y0,
        max_steps=None,
    )
    schedule_free_sgd_out = optx.minimise(
        coupled_rosenbrock,
        ScheduleFreeSGD(learning_rate, beta=0.99, rtol=rtol, atol=atol),
        y0,
        max_steps=None,
    )

    # Yes these tolerances are very high, doesn't really matter
    assert jnp.allclose(jnp.ones(100), gd_out.value, rtol=1e-2, atol=1e-3)
    assert jnp.allclose(
        jnp.ones(100), schedule_free_adam_out.value, rtol=1e-2, atol=1e-3
    )
    assert jnp.allclose(
        jnp.ones(100), schedule_free_sgd_out.value, rtol=1e-2, atol=1e-3
    )
