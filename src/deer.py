""" deer.py
Code adapted from the original DEER codebase by Lim et al. (2024): https://github.com/machine-discovery/deer 
Based on commit: 17b0b625d3413cb3251418980fb78916e5dacfaa (1/18/24) 
Copyright (c) 2023, Machine Discovery Ltd 
Licensed under the BSD 3-Clause License (see LICENSE file for details).

Modifications for benchmarking and quasi-DEER by Xavier Gonzalez (2024). """

from typing import Callable, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def _make_seq1d_progress_callback(
    max_iter: int, desc: str
) -> Tuple[Callable[[Any], None], Callable[[Any], None]]:
    """Host-side tqdm for ``jax.lax.while_loop`` Newton iters via ``jax.debug.callback``.

    Returns (on_step, on_close). ``on_close`` must run after the loop (ordered callback)
    so the bar finishes before host code runs (avoids a duplicate tqdm line after e.g. ``print``).
    """
    from tqdm import tqdm
    import sys

    pbar = tqdm(
        total=max_iter,
        desc=desc,
        unit="iter",
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.05,
    )

    def _on_step(step) -> None:
        s = int(np.asarray(step).item())
        pbar.n = min(s, max_iter)
        pbar.refresh()

    def _on_close(_token) -> None:
        pbar.close()

    return _on_step, _on_close


def seq1d(
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    y0: jnp.ndarray,
    xinp: Any,
    params: Any,
    yinit_guess: Optional[jnp.ndarray] = None,
    max_iter: int = 10000,
    memory_efficient: bool = False,
    quasi: bool = False,
    qmem_efficient: bool = True,  # XG addition
    full_trace: bool = False,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner,
    clip_val: float=1e8,
    show_progress: bool = False,
    tol: Optional[float] = None,
    rtol: Optional[float] = None,
):
    """
    Solve the discrete sequential equation, y[i + 1] = func(y[i], x[i], params) with the DEER framework.

    Arguments
    ---------
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray]
        The Markov transition kernel.
        The arguments are: the current state (D,), external input signal x (*nx,) in 
        a pytree (in this case the RNG source), and other parameters of the transition kernel. For example,
        for HMC, these parameters are the step size and the number of leapfrog steps. The return value 
        is the next state (D,).
    y0: jnp.ndarray
        Initial state of the chain (D,).
    xinp: Any
        The external input signal in a pytree of shape (T, *nx).
    params: Any
        The parameters of the function ``func``.
    yinit_guess: jnp.ndarray or None
        The initial guess of the full MCMC trajectory (T, D).
        If None, it will be initialized to the 0 vector.
    max_iter: int
        The maximum number of iterations to perform.
    memory_efficient: bool
        If True, then use the memory efficient algorithm for the DEER iteration.
    quasi: bool
        If True, then make all the Jacobians diagonal. (XG addition)
    qmem_efficient: bool
        If True, use the memory efficient of quasi; if false, call jnp.diag on jac func
        Note: need to use false for eigenworms
    full_trace: bool
        If True, return the full trace of all the Newton iterates for a fixed specification of max_iter (uses a scan)
        if False, return only the final iterate (uses a jax.lax.while_loop)
    show_progress: bool
        If True and ``full_trace`` is False, report Newton iteration progress on stderr (tqdm) during execution.
    tol: float or None
        Absolute tolerance for Newton early stopping (``err > tol``). If None, uses dtype defaults in the solver.
    rtol: float or None
        Relative scale for the error term. If None, uses dtype defaults in the solver.

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the discrete difference equation (nsamples, ny),
        excluding the initial states.
    """
    # set the default initial guess
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    if yinit_guess is None:
        yinit_guess = jnp.zeros(
            (xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype
        )  # (nsamples, ny)

    def shifter_func(y: jnp.ndarray, shifter_params: Any) -> jnp.ndarray:
        """
        Shift y to the left by one step such that y[i+1] = y[i] and y[0] = y0; takes (T, D) -> (T, D).
        """
        (y0,) = shifter_params
        y = jnp.concatenate((y0[None, :], y[:-1, :]), axis=0)  # (nsamples, ny)
        return y

    if quasi:
        yt, _, _, _, samp_iters = diagonal_deer_iteration_helper(
            inv_lin=diagonal_seq1d_inv_lin,
            func=func,
            shifter_func=shifter_func,
            params=params,
            xinput=xinp,
            inv_lin_params=(y0,),
            shifter_func_params=(y0,),
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=True,
            full_trace=full_trace,
            qmem_efficient=qmem_efficient,
            damp_factor=damp_factor,
            preconditioner=preconditioner,
            clip_val=clip_val,
            show_progress=show_progress,
            tol=tol,
            rtol=rtol,
        )
    else:
        yt, _, _, _, samp_iters = deer_iteration(
            inv_lin=seq1d_inv_lin,
            func=func,
            shifter_func=shifter_func,
            params=params,
            xinput=xinp,
            inv_lin_params=(y0,),
            shifter_func_params=(y0,),
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=True,
            full_trace=full_trace,
            damp_factor=damp_factor,
            preconditioner=preconditioner,
            clip_val=clip_val,
            show_progress=show_progress,
            tol=tol,
            rtol=rtol,
        )
    if full_trace:
        return (jnp.vstack((yinit_guess[None, ...], yt)), samp_iters)
    else:
        return (yt, samp_iters)


def deer_iteration(
    inv_lin: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], jnp.ndarray],
    params: Any,
    xinput: Any,
    inv_lin_params: Any,
    shifter_func_params: Any,
    yinit_guess: jnp.ndarray,
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    full_trace: bool = False,
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner for Quasi
    clip_val: float=1e8,
    show_progress: bool = False,
    tol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Callable]:

    progress_cb = None
    progress_close = None
    if show_progress and not full_trace:
        progress_cb, progress_close = _make_seq1d_progress_callback(
            max_iter, "DEER seq1d Newton"
        )

    jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = yinit_guess.dtype
    default_tol = 1e-7 if dtype == jnp.float64 else 1e-4
    default_rtol = 1e-4 if dtype == jnp.float64 else 1e-3
    tol_effective = default_tol if tol is None else tol
    rtol_effective = default_rtol if rtol is None else rtol

    def iter_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        err, yt, gt_, iiter = iter_inp
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        gt = -jnp.clip(damp_factor * jacfunc(ytparams, xinput, params), -clip_val, clip_val)
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)
        rhs += jnp.einsum("...ij,...j->...i", gt, ytparams)
        yt_next = inv_lin(gt, rhs, inv_lin_params)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, a_min=-clip, a_max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        err = jnp.max(jnp.abs(yt_next - yt) - rtol_effective * jnp.abs(yt))

        next_iiter = iiter + 1
        if progress_cb is not None:
            jax.debug.callback(progress_cb, next_iiter, ordered=True)
        return err, yt_next, gt, next_iiter

    def scan_func(iter_inp, args):
        err, yt, gt_, iiter = iter_inp
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        gt = -jnp.clip(damp_factor * jacfunc(ytparams, xinput, params), -clip_val, clip_val)
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)
        rhs += jnp.einsum("...ij,...j->...i", gt, ytparams)
        yt_next = inv_lin(gt, rhs, inv_lin_params)  # (nsamples, ny)

        err = jnp.max( jnp.abs(yt_next - yt) - rtol_effective * jnp.abs(yt) )

        yt_next = jnp.nan_to_num(yt_next)  # XG addition, avoid nans
        new_carry = err, yt_next, gt, iiter + 1
        return new_carry, yt_next

    def cond_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> bool:
        err, _, _, iiter = iter_inp
        return jnp.logical_and(err > tol_effective, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    gt = jnp.zeros(
        (yinit_guess.shape[0], yinit_guess.shape[-1], yinit_guess.shape[-1]),
        dtype=dtype,
    )

    iiter = jnp.array(0, dtype=jnp.int32)
    if full_trace:
        _, yt = jax.lax.scan(
            scan_func, (err, yinit_guess, gt, iiter), None, length=max_iter
        )
        samp_iters = max_iter
    else:
        _, yt, gt, samp_iters = jax.lax.while_loop(
            cond_func, iter_func, (err, yinit_guess, gt, iiter)
        )
    if progress_close is not None:
        jax.debug.callback(progress_close, samp_iters, ordered=True)
    if memory_efficient:
        gt = None
    rhs = jnp.zeros_like(gt[..., 0])  # (nsamples, ny)
    return yt, gt, rhs, func, samp_iters


def binary_operator(
    element_i: Tuple[jnp.ndarray, jnp.ndarray],
    element_j: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    a = gtj @ gti
    b = jnp.einsum("...ij,...j->...i", gtj, hti) + htj
    return a, b


def matmul_recursive(
    mats: jnp.ndarray, vecs: jnp.ndarray, y0: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve the linear recurrence y[t+1] = mats[t] @ y[t] + vecs[t] for all t in parallel.

    Composing two affine maps is itself an affine map: M_j @ (M_i @ y + v_i) + v_j = (M_j @ M_i) @ y + (M_j @ v_i + v_j).

    The initial condition y0 is encoded as the zeroth element (I, y0) so that all elements
    have the same shape as required by associative_scan. The output at index 0 recovers y0,
    and subsequent indices give y[1], y[2], ..., y[nsamples].

    Arguments
    ---------
    mats: jnp.ndarray
        The matrices to be multiplied, shape (nsamples - 1, ny, ny)
    vecs: jnp.ndarray
        The vector to be multiplied, shape (nsamples - 1, ny)
    y0: jnp.ndarray
        The initial condition, shape (ny,)

    Returns
    -------
    result: jnp.ndarray
        The result of the matrix multiplication, shape (nsamples, ny)
    """

    eye = jnp.eye(mats.shape[-1], dtype=mats.dtype)[None]  # (1, ny, ny)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (nsamples, ny, ny)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (nsamples, ny)

    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(binary_operator, elems)
    return yt  # (nsamples, ny)


def seq1d_inv_lin(
    gmat: jnp.ndarray, rhs: jnp.ndarray, inv_lin_params: Tuple[jnp.ndarray]
) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the discrete sequential equation.
    y[i + 1] + G[i] y[i] = rhs[i], y[0] = y0.

    Arguments
    ---------
    gmat: jnp.ndarray
        The G-matrix of shape (nsamples, ny, ny).
    rhs: jnp.ndarray
        The right hand side of the equation of shape (nsamples, ny).
    inv_lin_params: Tuple[jnp.ndarray]
        The parameters of the linear operator.
        The first element is the initial condition (ny,).

    Returns
    -------
    y: jnp.ndarray
        The solution of the linear equation of shape (nsamples, ny).
    """
    (y0,) = inv_lin_params

    # compute the recursive matrix multiplication and drop the first element
    yt = matmul_recursive(-gmat, rhs, y0)[1:]  # (nsamples, ny)
    return yt

# ---------------------------------------------------------------------------#
#                                Quasi
#                                  XG addition to do quasi-Newton (i.e. just use diagonalized Jacobians)
# ---------------------------------------------------------------------------#


def diagonal_binary_operator(
    element_i: Tuple[jnp.ndarray, jnp.ndarray],
    element_j: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    XG addition to make the matrix multiplication diagonal
    """
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    a = gtj * gti
    b = gtj * hti + htj
    return a, b


def diagonal_matmul_recursive(
    mats: jnp.ndarray, vecs: jnp.ndarray, y0: jnp.ndarray
) -> jnp.ndarray:
    """
    XG addition to make the matrix multiplication diagonal

    Perform the matrix multiplication recursively, y[i + 1] = mats[i] @ y[i] + vec[i].

    Arguments
    ---------
    mats: jnp.ndarray
        The matrices to be multiplied, shape (nsamples - 1, ny) # changed to make the matrices diagonal
    vecs: jnp.ndarray
        The vector to be multiplied, shape (nsamples - 1, ny)
    y0: jnp.ndarray
        The initial condition, shape (ny,)

    Returns
    -------
    result: jnp.ndarray
        The result of the matrix multiplication, shape (nsamples, ny)
    """
    # shift the elements by one index
    eye = jnp.ones(mats.shape[-1], dtype=mats.dtype)[None]  # (1, ny)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (nsamples, ny)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (nsamples, ny)

    # perform the scan
    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(diagonal_binary_operator, elems)
    return yt  # (nsamples, ny)


def diagonal_seq1d_inv_lin(
    gmat: jnp.ndarray, rhs: jnp.ndarray, inv_lin_params: Tuple[jnp.ndarray]
) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the discrete sequential equation.
    y[i + 1] + G[i] y[i] = rhs[i], y[0] = y0.

    Arguments
    ---------
    gmat: jnp.ndarray
        The diagonal G-matrix of shape (nsamples, ny). (XG addition)
    rhs: jnp.ndarray
        The right hand side of the equation of shape (nsamples, ny).
    inv_lin_params: Tuple[jnp.ndarray]
        The parameters of the linear operator.
        The first element is the initial condition (ny,).

    Returns
    -------
    y: jnp.ndarray
        The solution of the linear equation of shape (nsamples, ny).
    """
    (y0,) = inv_lin_params

    # compute the recursive matrix multiplication and drop the first element
    yt = diagonal_matmul_recursive(-gmat, rhs, y0)[1:]  # (nsamples, ny)
    return yt


def diagonal_deer_iteration_helper(
    inv_lin: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], jnp.ndarray],
    params: Any,  # gradable
    xinput: Any,  # gradable
    inv_lin_params: Any,  # gradable
    shifter_func_params: Any,  # gradable
    yinit_guess: jnp.ndarray,
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    full_trace: bool = False,  # XG addition
    qmem_efficient: bool = True,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner
    clip_val: float=1e8,
    show_progress: bool = False,
    tol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Callable]:
    progress_cb = None
    progress_close = None
    if show_progress and not full_trace:
        progress_cb, progress_close = _make_seq1d_progress_callback(
            max_iter, "DEER seq1d Newton (quasi)"
        )

    jacfunc = jax.vmap(
        jax.jacfwd(func, argnums=0), in_axes=(0, 0, None)
    )  # bunch of dense matrices

    precond = preconditioner if preconditioner is not None else jnp.ones((yinit_guess.shape[-1]))

    if qmem_efficient:
        def deer_jvp(z, driver, params, v):
            return jax.jvp(lambda z : func(z, driver, params), (z,), (v, ))[1]
        keys = jr.split(params['key'], (xinput.shape[0]))

    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = yinit_guess.dtype
    default_tol = 1e-7 if dtype == jnp.float64 else 5e-4
    default_rtol = 1e-4 if dtype == jnp.float64 else 1e-3
    tol_effective = default_tol if tol is None else tol
    rtol_effective = default_rtol if rtol is None else rtol

    def iter_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        err, yt, gt_, iiter = iter_inp
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        if qmem_efficient:
            gt = -jnp.clip(
                damp_factor / precond[None,:] * jax.vmap(quasi_diag_estimator, in_axes=(0, 0, None, None, 0))(
                    ytparams, xinput, params, deer_jvp, keys),
                -clip_val, clip_val,
            )
        else:
            gt = -jnp.clip(
                damp_factor / precond[None,:] * jax.vmap(jnp.diag)(jacfunc(ytparams, xinput, params)),
                -clip_val, clip_val,
            )
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)
        rhs += gt * ytparams
        yt_next = inv_lin(gt, rhs, inv_lin_params)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, a_min=-clip, a_max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        err = jnp.max( jnp.abs(yt_next - yt) - rtol_effective * jnp.abs(yt) )

        next_iiter = iiter + 1
        if progress_cb is not None:
            jax.debug.callback(progress_cb, next_iiter, ordered=True)
        return err, yt_next, gt, next_iiter

    def scan_func(iter_inp, args):
        err, yt, gt_, iiter = iter_inp
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        if qmem_efficient:
            gt = -jnp.clip(
                damp_factor / precond[None,:] * jax.vmap(quasi_diag_estimator, in_axes=(0, 0, None, None, 0))(
                    ytparams, xinput, params, deer_jvp, keys),
                -clip_val, clip_val,
            )
        else:
            gt = -jnp.clip(
                damp_factor / precond[None,:] * jax.vmap(jnp.diag)(jacfunc(ytparams, xinput, params)),
                -clip_val, clip_val,
            )
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)
        rhs += gt * ytparams
        yt_next = inv_lin(gt, rhs, inv_lin_params)  # (nsamples, ny)

        err = jnp.max( jnp.abs(yt_next - yt) - rtol_effective * jnp.abs(yt) )

        yt_next = jnp.nan_to_num(yt_next)  # XG addition, avoid nans
        new_carry = err, yt_next, gt, iiter + 1
        return new_carry, yt_next

    def cond_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> bool:
        err, _, _, iiter = iter_inp
        return jnp.logical_and(err > tol_effective, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    gt = jnp.zeros(
        (yinit_guess.shape[0], yinit_guess.shape[-1]),
        dtype=dtype,
    )
    iiter = jnp.array(0, dtype=jnp.int32)
    if full_trace:
        _, yt = jax.lax.scan(
            scan_func, (err, yinit_guess, gt, iiter), None, length=max_iter
        )
        samp_iters = max_iter
    else:
        _, yt, gt, samp_iters = jax.lax.while_loop(
            cond_func, iter_func, (err, yinit_guess, gt, iiter)
        )
    if progress_close is not None:
        jax.debug.callback(progress_close, samp_iters, ordered=True)
    if memory_efficient:
        gt = None
    rhs = jnp.zeros_like(gt)  # (nsamples, ny)
    return yt, gt, rhs, func, samp_iters

def quasi_diag_estimator(state, inputs, params, deer_jvp, key, num_samples=1):
    z_rad = jr.rademacher(key, (num_samples, state.shape[0])).astype(float)
    vmap_jvp = jax.vmap(deer_jvp, in_axes=(None, None, None, 0))
    jac_diag = jnp.mean(z_rad * vmap_jvp(state, inputs, params, z_rad), axis=0)
    return jac_diag
