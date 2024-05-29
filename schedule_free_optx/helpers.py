from collections.abc import Callable
from typing_extensions import TypeAlias
from typing import Any, Literal, Union, overload, TypeVar

import equinox as eqx
import equinox.internal as eqxi
from equinox.internal import ω
import optimistix as optx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool, ArrayLike, PyTree, Scalar

Y = TypeVar("Y")
Aux = TypeVar("Aux")
Args: TypeAlias = Any
Out = TypeVar("Out")
SearchState = TypeVar("SearchState")
DescentState = TypeVar("DescentState")
Fn: TypeAlias = Callable[[Y, Args], tuple[Out, Aux]]
_F = TypeVar("_F")

GradFnInfo: TypeAlias = Union[
    optx.FunctionInfo.EvalGrad,
    optx.FunctionInfo.EvalGradHessian,
    optx.FunctionInfo.EvalGradHessianInv,
    optx.FunctionInfo.ResidualJac,
]


def tree_where(
    pred: Bool[ArrayLike, ""], true: PyTree[ArrayLike], false: PyTree[ArrayLike]
) -> PyTree[Array]:
    """Return the `true` or `false` pytree depending on `pred`."""
    keep = lambda a, b: jnp.where(pred, a, b)
    return jtu.tree_map(keep, true, false)


def cauchy_termination(
    rtol: float,
    atol: float,
    norm: Callable[[PyTree], Scalar],
    y: Y,
    y_diff: Y,
    f: _F,
    f_diff: _F,
) -> Bool[Array, ""]:
    """Terminate if there is a small difference in both `y` space and `f` space, as
    determined by `rtol` and `atol`.

    Specifically, this checks that `y_diff < atol + rtol * y` and
    `f_diff < atol + rtol * f_prev`, terminating if both of these are true.
    """
    y_scale = (atol + rtol * ω(y).call(jnp.abs)).ω
    f_scale = (atol + rtol * ω(f).call(jnp.abs)).ω
    y_converged = norm((ω(y_diff).call(jnp.abs) / y_scale**ω).ω) < 1
    f_converged = norm((ω(f_diff).call(jnp.abs) / f_scale**ω).ω) < 1
    return y_converged & f_converged


@overload
def tree_full_like(
    struct: PyTree[Union[Array, jax.ShapeDtypeStruct]],
    fill_value: ArrayLike,
    allow_static: Literal[False] = False,
):
    ...


@overload
def tree_full_like(
    struct: PyTree, fill_value: ArrayLike, allow_static: Literal[True] = True
):
    ...


def tree_full_like(struct: PyTree, fill_value: ArrayLike, allow_static: bool = False):
    """Return a pytree with the same type and shape as the input with values
    `fill_value`.

    If `allow_static=True`, then any non-{array, struct}s are ignored and left alone.
    If `allow_static=False` then any non-{array, struct}s will result in an error.
    """
    fn = lambda x: jnp.full(x.shape, fill_value, x.dtype)
    if isinstance(fill_value, (int, float)):
        if fill_value == 0:
            fn = lambda x: jnp.zeros(x.shape, x.dtype)
        elif fill_value == 1:
            fn = lambda x: jnp.ones(x.shape, x.dtype)
    if allow_static:
        _fn = fn
        fn = (
            lambda x: _fn(x)
            if eqx.is_array(x) or isinstance(x, jax.ShapeDtypeStruct)
            else x
        )
    return jtu.tree_map(fn, struct)


def tree_zeros_like(struct: PyTree):
    return tree_full_like(struct, jnp.asarray(0.0))


def lin_to_grad(lin_fn, *primals):
    return jax.linear_transpose(lin_fn, *primals)(1.0)


def filter_cond(pred, true_fun, false_fun, *operands):
    dynamic, static = eqx.partition(operands, eqx.is_array)

    def _true_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = true_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    def _false_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = false_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    dynamic_out, static_out = lax.cond(pred, _true_fun, _false_fun, dynamic)
    return eqx.combine(dynamic_out, static_out.value)
