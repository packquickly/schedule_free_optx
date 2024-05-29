from collections.abc import Callable
from typing import Any, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .helpers import (
    Aux,
    cauchy_termination,
    Fn,
    Out,
    SearchState,
    tree_zeros_like,
    Y,
)


#
# NOTE: This solver is hardcoded and does not rely on the Optimistix abstractions
# of `search` and `descent`. This is not for a technical reason other than it requires
# a new top-level solver which is more difficult to write, and this is so niche
# that it's not worth the extra effort to do so. Nothing in Optimistix
# forces us to use the `search` and `descent` paradigm so in this case I just chose to
# avoid it.
#


class _SFState(eqx.Module, Generic[Y, Out, Aux, SearchState]):
    grad_eval_point: Y
    z: Y
    var_estimate: Y
    sum_of_step_sizes: Scalar
    f_val: Scalar
    aux: Aux
    terminate: Bool[Array, ""]
    result: optx.RESULTS
    step: Scalar


class ScheduleFreeAdamW(optx.AbstractMinimiser[Y, Aux, _SFState]):
    learning_rate: float
    beta_1: float
    beta_2: float
    warmup_steps: int
    weight_decay: float
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    epsilon: float

    def __init__(
        self,
        learning_rate: float,
        beta_1: float,
        beta_2: float,
        warmup_steps: int,
        weight_decay: float,
        rtol: float,
        atol: float,
        norm: Callable = optx.max_norm,
        epsilon: float = 1e-8,
    ):
        if beta_1 > 1 or beta_1 < 0:
            raise ValueError("`beta_1` must be in the range [0,1].")
        if beta_2 > 1 or beta_2 < 0:
            raise ValueError("`beta_2` must be in the range [0,1].")
        if learning_rate <= 0:
            raise ValueError("`learning_rate` must be greater than 0.")
        if warmup_steps <= 0:
            raise ValueError("`warmup_steps` must be greater than 0.")
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.epsilon = epsilon

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _SFState:
        return _SFState(
            grad_eval_point=tree_zeros_like(y),
            z=y,
            var_estimate=tree_zeros_like(y),
            sum_of_step_sizes=jnp.asarray(0.0),
            f_val=jnp.asarray(jnp.inf),
            aux=tree_zeros_like(aux_struct),
            terminate=jnp.array(False),
            result=optx.RESULTS.successful,
            step=jnp.asarray(1),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _SFState,
        tags: frozenset[object],
    ) -> tuple[Y, _SFState, Aux]:
        # `f_eval` is evaluated at a point different than the current
        # optimum estimate.
        new_grad_eval_point = ((1 - self.beta_1) * state.z**ω + self.beta_1 * y**ω).ω
        (f_eval, aux_eval), grad = jax.value_and_grad(
            lambda _y: fn(_y, args), has_aux=True
        )(new_grad_eval_point)
        var = jtu.tree_map(jnp.square, grad)
        new_var_estimate = (
            self.beta_2 * state.var_estimate**ω + (1 - self.beta_2) * var**ω
        ).ω
        unbiased_var = ((new_var_estimate**ω) / (1 - self.beta_2**state.step)).ω
        step_size = self.learning_rate * jnp.minimum(1, state.step / self.warmup_steps)
        new_z = (
            state.z**ω
            - (step_size * (grad**ω) / (ω(unbiased_var).call(jnp.sqrt) + self.epsilon))
            - step_size * self.weight_decay * new_grad_eval_point**ω
        ).ω
        new_sum_of_stepsizes = state.sum_of_step_sizes + step_size**2
        c = (step_size**2) / new_sum_of_stepsizes
        new_y = ((1 - c) * y**ω + c * new_z**ω).ω

        y_diff = (new_y**ω - y**ω).ω
        f_diff = f_eval - state.f_val

        # WARNING: this termination condition uses the proper `y` in `y`-space, but uses
        # the function values associated with `new_grad_eval_point` in `f`-space.
        # My assumption is this shouldn't matter much, as I expect most users to use
        # this solver step-by-step for online optimisation and not use the termination
        # condition at all.
        terminate = cauchy_termination(
            self.rtol, self.atol, self.norm, y, y_diff, f_eval, f_diff
        )

        new_state = _SFState(
            new_grad_eval_point,
            new_z,
            new_var_estimate,
            new_sum_of_stepsizes,
            f_eval,
            aux_eval,
            terminate,
            optx.RESULTS.successful,
            state.step + 1,
        )

        return new_y, new_state, aux_eval

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _SFState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], optx.RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _SFState,
        tags: frozenset[object],
        result: optx.RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


ScheduleFreeAdamW.__init__.__doc__ = """**Arguments:**

    - `learning_rate`: Specifies a constant learning rate to use at each step.
    - `beta_1`: the constant `beta_1` which determines how much to average the new step
        when computing the gradient.
    - `beta_2`: the constant `beta_2` for exponential moving avarage analogous to Adam.
    - `warmup_steps`: The number of warmup steps to perform to get to the full learning
        rate.
    - `weight_decay`: Specifies the amount of weight decay.
    - `rtol`: Relative tolerance for terminating the solve.
    - `atol`: Absolute tolerance for terminating the solve.
    - `norm`: The norm used to determine the difference between two iterates in the
        convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
        includes three built-in norms: [`optimistix.max_norm`][],
        [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
    - `epsilon`: the constant `epsilon` for avoiding division by 0 in Adam
    """
