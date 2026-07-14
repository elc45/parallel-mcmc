""" deer.py
Code adapted from the original DEER codebase by Lim et al. (2024): https://github.com/machine-discovery/deer 
Based on commit: 17b0b625d3413cb3251418980fb78916e5dacfaa (1/18/24) 
Copyright (c) 2023, Machine Discovery Ltd 
Licensed under the BSD 3-Clause License (see LICENSE file for details).

Modifications for benchmarking and quasi-DEER by Xavier Gonzalez (2024). """

from typing import Callable, Any, Tuple, Optional, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr


class NewtonHistories(NamedTuple):
    """Scalar Newton diagnostics collected without storing full state traces.

    ``newton_err`` has length ``max_iter`` (valid prefix ``[:samp_iters]``).
    ``residual_sq`` has length ``max_iter + 1`` (valid prefix ``[:samp_iters + 1]``),
    where index ``0`` is the initial guess (matching ``full_trace=True`` layout).
    """

    newton_err: jnp.ndarray
    residual_sq: jnp.ndarray


def seq1d(
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    y0: jnp.ndarray,
    xinp: Any,
    params: Any,
    init_trajectory_guess: Optional[jnp.ndarray] = None,
    max_iter: int = 10000,
    memory_efficient: bool = False,
    quasi: bool = False,
    qmem_efficient: bool = True,  # XG addition
    full_trace: bool = False,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner,
    clip_val: float=1e8,
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
    init_trajectory_guess: jnp.ndarray or None
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
    tol: float or None
        Absolute tolerance for Newton early stopping (``err > tol``). If None, uses dtype defaults in the solver.
    rtol: float or None
        Relative scale for the error term. If None, uses dtype defaults in the solver.

    Returns
    -------
    y: jnp.ndarray
        With ``full_trace=False``: final trajectory ``(T, D)``.
        With ``full_trace=True``: Newton trace ``(max_iter + 1, T, D)`` including the initial guess.
    samp_iters: jnp.ndarray
        Number of Newton iterations performed (or first-converged index for ``full_trace=True``).
    newton_hist: NewtonHistories or None
        With ``full_trace=False``: per-iteration Newton error and fixed-point residual scalars.
        With ``full_trace=True``: ``None`` (reconstruct from the returned state trace instead).
    """
    # set the default initial guess
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    if init_trajectory_guess is None:
        init_trajectory_guess = jnp.zeros(
            (xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype
        )  # (T, D)

    def shifter_func(y: jnp.ndarray, shifter_params: Any) -> jnp.ndarray:
        """
        Shift y to the left by one step such that y[i+1] = y[i] and y[0] = y0; takes (T, D) -> (T, D).
        """
        (y0,) = shifter_params
        y = jnp.concatenate((y0[None, :], y[:-1, :]), axis=0)  # (nsamples, ny)
        return y

    if quasi:
        yt, _, _, _, samp_iters, newton_hist = diagonal_deer_iteration(
            inv_lin=diagonal_seq1d_inv_lin,
            func=func,
            shifter_func=shifter_func,
            params=params,
            xinput=xinp,
            init=y0,
            shifter_func_params=(y0,),
            init_trajectory_guess=init_trajectory_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=True,
            full_trace=full_trace,
            qmem_efficient=qmem_efficient,
            damp_factor=damp_factor,
            preconditioner=preconditioner,
            clip_val=clip_val,
            tol=tol,
            rtol=rtol,
        )
    else:
        yt, samp_iters, newton_hist = deer_iteration(
            inv_lin=seq1d_inv_lin,
            func=func,
            shifter_func=shifter_func,
            params=params,
            xinput=xinp,
            init=y0,
            shifter_func_params=(y0,),
            init_trajectory_guess=init_trajectory_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=True,
            full_trace=full_trace,
            damp_factor=damp_factor,
            preconditioner=preconditioner,
            clip_val=clip_val,
            tol=tol,
            rtol=rtol,
        )
    if full_trace:
        return (jnp.vstack((init_trajectory_guess[None, ...], yt)), samp_iters, None)
    else:
        return (yt, samp_iters, newton_hist)


def deer_iteration(
    inv_lin: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], jnp.ndarray],
    params: Any,
    xinput: Any,
    init: Any,
    shifter_func_params: Any,
    init_trajectory_guess: jnp.ndarray,
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    full_trace: bool = False,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner
    clip_val: float=1e8,
    tol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[NewtonHistories]]:
    yt, _, _, _, samp_iters, newton_hist = deer_iteration_helper(
        inv_lin=inv_lin,
        func=func,
        shifter_func=shifter_func,
        params=params,
        xinput=xinput,
        inv_lin_params=(init,),
        shifter_func_params=shifter_func_params,
        init_trajectory_guess=init_trajectory_guess,
        max_iter=max_iter,
        memory_efficient=memory_efficient,
        clip_ytnext=clip_ytnext,
        full_trace=full_trace,
        damp_factor=damp_factor,
        preconditioner=preconditioner, 
        clip_val=clip_val,
        tol=tol,
        rtol=rtol,
    )
    return (yt, samp_iters, newton_hist)


def deer_iteration_helper(
    inv_lin: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], jnp.ndarray],
    params: Any,  # gradable
    xinput: Any,  # gradable
    inv_lin_params: Any,  # gradable
    shifter_func_params: Any,  # gradable
    init_trajectory_guess: jnp.ndarray,
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    full_trace: bool = False,
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner for Quasi
    clip_val: float=1e8,
    tol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray, Callable, jnp.ndarray, Optional[NewtonHistories]]:

    jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = init_trajectory_guess.dtype
    default_tol = 1e-7 if dtype == jnp.float64 else 1e-4
    default_rtol = 1e-4 if dtype == jnp.float64 else 1e-3
    tol_effective = default_tol if tol is None else tol
    rtol_effective = default_rtol if rtol is None else rtol

    def iter_func(iter_inp):
        err, Y_i, gt_, iiter, err_hist, res_hist = iter_inp
        # Y_i: (T, D) — full trajectory at Newton iterate i
        Y_i_shifted = shifter_func(Y_i, shifter_func_params)
        gt = -jnp.clip(damp_factor * jacfunc(Y_i_shifted, xinput, params), -clip_val, clip_val)
        f_vals = func2(Y_i_shifted, xinput, params)
        res_hist = res_hist.at[iiter].set(jnp.sum(jnp.square(Y_i - f_vals)))
        rhs = f_vals + jnp.einsum("...ij,...j->...i", gt, Y_i_shifted)
        Y_i_next = inv_lin(gt, rhs, inv_lin_params)  # (T, D)

        if clip_ytnext:
            clip = 1e8
            Y_i_next = jnp.clip(Y_i_next, a_min=-clip, a_max=clip)
            Y_i_next = jnp.where(jnp.isnan(Y_i_next), 0.0, Y_i_next)

        err = jnp.max(jnp.abs(Y_i_next - Y_i) - rtol_effective * jnp.abs(Y_i))
        err_hist = err_hist.at[iiter].set(err)

        return err, Y_i_next, gt, iiter + 1, err_hist, res_hist

    def scan_func(iter_inp, args):
        err, Y_i, gt_, iiter, converged, conv_iter = iter_inp

        def do_work(Y_i_gt):
            Y_i_in, gt_in = Y_i_gt
            # Y_i_in: (T, D) — full trajectory at Newton iterate i
            Y_i_shifted = shifter_func(Y_i_in, shifter_func_params)
            gt = -jnp.clip(damp_factor * jacfunc(Y_i_shifted, xinput, params), -clip_val, clip_val)
            rhs = func2(Y_i_shifted, xinput, params)
            rhs += jnp.einsum("...ij,...j->...i", gt, Y_i_shifted)
            Y_i_next = inv_lin(gt, rhs, inv_lin_params)
            err_new = jnp.max(jnp.abs(Y_i_next - Y_i_in) - rtol_effective * jnp.abs(Y_i_in))
            Y_i_next = jnp.nan_to_num(Y_i_next)
            return Y_i_next, gt, err_new

        def skip_work(Y_i_gt):
            Y_i_in, gt_in = Y_i_gt
            return Y_i_in, gt_in, err

        Y_i_next, gt_new, err_new = jax.lax.cond(converged, skip_work, do_work, (Y_i, gt_))

        newly_conv = (~converged) & (err_new <= tol_effective)
        converged_new = converged | newly_conv
        conv_iter_new = jnp.where(newly_conv, iiter + 1, conv_iter)

        new_carry = err_new, Y_i_next, gt_new, iiter + 1, converged_new, conv_iter_new
        return new_carry, Y_i_next

    def cond_func(iter_inp) -> bool:
        err, _, _, iiter, _, _ = iter_inp
        return jnp.logical_and(err > tol_effective, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    gt = jnp.zeros(
        (init_trajectory_guess.shape[0], init_trajectory_guess.shape[-1], init_trajectory_guess.shape[-1]),
        dtype=dtype,
    )

    iiter = jnp.array(0, dtype=jnp.int32)
    newton_hist: Optional[NewtonHistories] = None
    if full_trace:
        converged_init = jnp.array(False)
        conv_iter_init = jnp.array(max_iter, dtype=jnp.int32)
        (_, _, _, _, _, samp_iters), Y_i = jax.lax.scan(
            scan_func,
            (err, init_trajectory_guess, gt, iiter, converged_init, conv_iter_init),
            None,
            length=max_iter,
        )
    else:
        err_hist = jnp.zeros((max_iter,), dtype=dtype)
        res_hist = jnp.zeros((max_iter + 1,), dtype=dtype)
        _, Y_i, gt, samp_iters, err_hist, res_hist = jax.lax.while_loop(
            cond_func,
            iter_func,
            (err, init_trajectory_guess, gt, iiter, err_hist, res_hist),
        )
        Y_final_shifted = shifter_func(Y_i, shifter_func_params)
        res_final = jnp.sum(jnp.square(Y_i - func2(Y_final_shifted, xinput, params)))
        res_hist = res_hist.at[samp_iters].set(res_final)
        newton_hist = NewtonHistories(newton_err=err_hist, residual_sq=res_hist)
    rhs = jnp.zeros_like(gt[..., 0])  # (T, D)
    if memory_efficient:
        gt = None
    return Y_i, gt, rhs, func, samp_iters, newton_hist


def binary_operator(
    element_i: Tuple[jnp.ndarray, jnp.ndarray],
    element_j: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    The associative binary operator: 
    (F_j ∘ F_i)(y) = M_j @ (M_i @ y + v_i) + v_j = (M_j @ M_i) @ y + (M_j @ v_i + v_j).
    """
    M_i, v_i = element_i
    M_j, v_j = element_j
    a = M_j @ M_i
    b = jnp.einsum("...ij,...j->...i", M_j, v_i) + v_j
    return a, b


def matmul_recursive(
    mats: jnp.ndarray, 
    vecs: jnp.ndarray, 
    y0: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve the linear recurrence y[t+1] = mats[t] @ y[t] + vecs[t] for all t in parallel.

    The initial condition y0 is encoded as the zeroth element (I, y0) so that all elements
    have the same shape as required by associative_scan. The output at index 0 recovers y0,
    and subsequent indices give y[1], y[2], ..., y[T].

    Arguments
    ---------
    mats: jnp.ndarray
        The matrices to be multiplied, shape (T - 1, D, D)
    vecs: jnp.ndarray
        The vector to be multiplied, shape (T - 1, D)
    y0: jnp.ndarray
        The initial condition, shape (D,)

    Returns
    -------
    result: jnp.ndarray
        The result of the matrix multiplication, shape (T, D)
    """

    eye = jnp.eye(mats.shape[-1], dtype=mats.dtype)[None]  # (1, D, D)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (T, D, D)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (T, D)

    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(binary_operator, elems)
    return yt


def seq1d_inv_lin(
    gmat: jnp.ndarray, rhs: jnp.ndarray, inv_lin_params: Tuple[jnp.ndarray]
) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the discrete sequential equation.
    y[i + 1] + G[i] y[i] = rhs[i], y[0] = init.

    Arguments
    ---------
    gmat: jnp.ndarray
        The G-matrix of shape (nsamples, ny, ny).
    rhs: jnp.ndarray
        The right hand side of the equation of shape (T, D).
    init: jnp.ndarray
        Initial condition (D,) for the linear recurrence.

    Returns
    -------
    y: jnp.ndarray
        The solution of the linear equation of shape (T, D).
    """
    (y0,) = inv_lin_params

    # compute the recursive matrix multiplication and drop the first element
    yt = matmul_recursive(-gmat, rhs, y0)[1:]
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
        The matrices to be multiplied, shape (T - 1, D) # changed to make the matrices diagonal
    vecs: jnp.ndarray
        The vector to be multiplied, shape (T - 1, D)
    y0: jnp.ndarray
        The initial condition, shape (D,)

    Returns
    -------
    result: jnp.ndarray
        The result of the matrix multiplication, shape (T, D)
    """
    # shift the elements by one index
    eye = jnp.ones(mats.shape[-1], dtype=mats.dtype)[None]  # (1, D)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (T, D)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (T, D)

    # perform the scan
    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(diagonal_binary_operator, elems)
    return yt  # (T, D)


def diagonal_seq1d_inv_lin(
    gmat: jnp.ndarray, rhs: jnp.ndarray, inv_lin_params: Tuple[jnp.ndarray]
) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the discrete sequential equation.
    y[i + 1] + G[i] y[i] = rhs[i], y[0] = init.

    Arguments
    ---------
    gmat: jnp.ndarray
        The diagonal G-matrix of shape (nsamples, ny). (XG addition)
    rhs: jnp.ndarray
        The right hand side of the equation of shape (T, D).
    init: jnp.ndarray
        Initial condition (D,) for the linear recurrence.

    Returns
    -------
    y: jnp.ndarray
        The solution of the linear equation of shape (T, D).
    """
    (y0,) = inv_lin_params

    # compute the recursive matrix multiplication and drop the first element
    yt = diagonal_matmul_recursive(-gmat, rhs, y0)[1:]  # (T, D)
    return yt


def diagonal_deer_iteration(
    inv_lin: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], jnp.ndarray],
    params: Any,  # gradable
    xinput: Any,  # gradable
    init: jnp.ndarray,  # gradable
    shifter_func_params: Any,  # gradable
    init_trajectory_guess: jnp.ndarray,
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    full_trace: bool = False,  # XG addition
    qmem_efficient: bool = True,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner
    clip_val: float=1e8,
    tol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray, Callable, jnp.ndarray, Optional[NewtonHistories]]:
    inv_lin_params = (init,)

    jacfunc = jax.vmap(
        jax.jacfwd(func, argnums=0), in_axes=(0, 0, None)
    )  # bunch of dense matrices

    precond = preconditioner if preconditioner is not None else jnp.ones((init_trajectory_guess.shape[-1]))

    if qmem_efficient:
        def deer_jvp(z, driver, params, v):
            return jax.jvp(lambda z : func(z, driver, params), (z,), (v, ))[1]
        # xinput can be an array or a pytree of arrays (e.g. NUTS drivers);
        # chain length is always the leading axis of the trajectory guess.
        keys = jr.split(params['key'], (init_trajectory_guess.shape[0],))

    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = init_trajectory_guess.dtype
    default_tol = 1e-7 if dtype == jnp.float64 else 5e-4
    default_rtol = 1e-4 if dtype == jnp.float64 else 1e-3
    tol_effective = default_tol if tol is None else tol
    rtol_effective = default_rtol if rtol is None else rtol

    def iter_func(iter_inp):
        err, yt, gt_, iiter, err_hist, res_hist = iter_inp
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
        f_vals = func2(ytparams, xinput, params)
        res_hist = res_hist.at[iiter].set(jnp.sum(jnp.square(yt - f_vals)))
        rhs = f_vals + gt * ytparams
        yt_next = inv_lin(gt, rhs, inv_lin_params)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, a_min=-clip, a_max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        err = jnp.max( jnp.abs(yt_next - yt) - rtol_effective * jnp.abs(yt) )
        err_hist = err_hist.at[iiter].set(err)

        return err, yt_next, gt, iiter + 1, err_hist, res_hist

    def scan_func(iter_inp, args):
        err, yt, gt_, iiter, converged, conv_iter = iter_inp

        def do_work(yt_gt):
            yt_in, gt_in = yt_gt
            # yt_in: (nsamples, ny)
            ytparams = shifter_func(yt_in, shifter_func_params)
            if qmem_efficient:
                gt = -jnp.clip(
                    damp_factor / precond[None, :] * jax.vmap(quasi_diag_estimator, in_axes=(0, 0, None, None, 0))(
                        ytparams, xinput, params, deer_jvp, keys
                    ),
                    -clip_val, clip_val,
                )
            else:
                gt = -jnp.clip(
                    damp_factor / precond[None, :] * jax.vmap(jnp.diag)(jacfunc(ytparams, xinput, params)),
                    -clip_val, clip_val,
                )
            rhs = func2(ytparams, xinput, params)
            rhs += gt * ytparams
            yt_next = inv_lin(gt, rhs, inv_lin_params)
            err_new = jnp.max(jnp.abs(yt_next - yt_in) - rtol_effective * jnp.abs(yt_in))
            yt_next = jnp.nan_to_num(yt_next)
            return yt_next, gt, err_new

        def skip_work(yt_gt):
            yt_in, gt_in = yt_gt
            return yt_in, gt_in, err

        yt_next, gt_new, err_new = jax.lax.cond(converged, skip_work, do_work, (yt, gt_))

        newly_conv = (~converged) & (err_new <= tol_effective)
        converged_new = converged | newly_conv
        conv_iter_new = jnp.where(newly_conv, iiter + 1, conv_iter)

        new_carry = err_new, yt_next, gt_new, iiter + 1, converged_new, conv_iter_new
        return new_carry, yt_next

    def cond_func(iter_inp) -> bool:
        err, _, _, iiter, _, _ = iter_inp
        return jnp.logical_and(err > tol_effective, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    gt = jnp.zeros(
        (init_trajectory_guess.shape[0], init_trajectory_guess.shape[-1]),
        dtype=dtype,
    )
    iiter = jnp.array(0, dtype=jnp.int32)
    newton_hist: Optional[NewtonHistories] = None
    if full_trace:
        converged_init = jnp.array(False)
        conv_iter_init = jnp.array(max_iter, dtype=jnp.int32)
        (_, _, _, _, _, samp_iters), yt = jax.lax.scan(
            scan_func,
            (err, init_trajectory_guess, gt, iiter, converged_init, conv_iter_init),
            None,
            length=max_iter,
        )
    else:
        err_hist = jnp.zeros((max_iter,), dtype=dtype)
        res_hist = jnp.zeros((max_iter + 1,), dtype=dtype)
        _, yt, gt, samp_iters, err_hist, res_hist = jax.lax.while_loop(
            cond_func,
            iter_func,
            (err, init_trajectory_guess, gt, iiter, err_hist, res_hist),
        )
        ytparams_final = shifter_func(yt, shifter_func_params)
        res_final = jnp.sum(jnp.square(yt - func2(ytparams_final, xinput, params)))
        res_hist = res_hist.at[samp_iters].set(res_final)
        newton_hist = NewtonHistories(newton_err=err_hist, residual_sq=res_hist)
    rhs = jnp.zeros_like(gt)  # (nsamples, ny)
    if memory_efficient:
        gt = None
    return yt, gt, rhs, func, samp_iters, newton_hist

def quasi_diag_estimator(state, inputs, params, deer_jvp, key, num_samples=1):
    z_rad = jr.rademacher(key, (num_samples, state.shape[0])).astype(float)
    vmap_jvp = jax.vmap(deer_jvp, in_axes=(None, None, None, 0))
    jac_diag = jnp.mean(z_rad * vmap_jvp(state, inputs, params, z_rad), axis=0)
    return jac_diag
