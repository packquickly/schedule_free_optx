from collections.abc import Callable
from typing import Any, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
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
    z: Y
    f_val: Scalar
    aux: Aux
    terminate: Bool[Array, ""]
    result: optx.RESULTS
    step: Scalar


class ScheduleFreeSGD(optx.AbstractMinimiser[Y, Aux, _SFState]):
    learning_rate: float
    beta: float
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]

    def __init__(
        self,
        learning_rate: float,
        beta: float,
        rtol: float,
        atol: float,
        norm: Callable = optx.max_norm,
    ):
        if beta > 1 or beta < 0:
            raise ValueError("`beta` must be in the range [0,1].")
        if learning_rate <= 0:
            raise ValueError("`learning_rate` must be greater than 0.")
        self.learning_rate = learning_rate
        self.beta = beta
        self.rtol = rtol
        self.atol = atol
        self.norm = norm

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
            z=y,
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
        grad_eval_point = ((1 - self.beta) * state.z**ω + self.beta * y**ω).ω
        (f_eval, aux_eval), grad = jax.value_and_grad(
            lambda _y: fn(_y, args), has_aux=True
        )(grad_eval_point)
        new_z = (state.z**ω - self.learning_rate * grad**ω).ω
        c = 1 / (state.step + 1)
        new_y = ((1 - c) * y**ω + c * new_z**ω).ω
        y_diff = (new_y**ω - y**ω).ω
        f_diff = f_eval - state.f_val

        terminate = cauchy_termination(
            self.rtol, self.atol, self.norm, y, y_diff, f_eval, f_diff
        )

        new_state = _SFState(
            new_z,
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


ScheduleFreeSGD.__init__.__doc__ = """**Arguments:**

    - `beta`: the constant `beta` which determines how much to average the new step when
        computing the gradient.
    - `learning_rate`: Specifies a constant learning rate to use at each step.
    - `rtol`: Relative tolerance for terminating the solve.
    - `atol`: Absolute tolerance for terminating the solve.
    - `norm`: The norm used to determine the difference between two iterates in the
        convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
        includes three built-in norms: [`optimistix.max_norm`][],
        [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
    """
