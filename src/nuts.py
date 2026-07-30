# Copyright 2020- The Blackjax Authors.
# Modifications for stop-gradient softmax proposal averaging.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""NUTS kernel with stop-gradient softmax proposal averaging.

Forward pass matches BlackJAX multinomial NUTS (discrete progressive sampling).
Backward pass differentiates a softmax-weighted average of every integrator
state encountered while building the orbit, with weights proportional to
``exp(-H(theta, momentum))`` (equivalently softmax of the per-state NUTS
proposal weights ``H0 - H``).

The blend uses the same ``stop_gradient`` trick as :func:`src.samplers.sigmoid_accept`:
on the forward pass the discrete sample is returned unchanged; on the backward
pass gradients flow through the soft average.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.termination as termination
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import IntegratorState
from blackjax.mcmc.proposal import (
    Proposal,
    progressive_biased_sampling,
    progressive_uniform_sampling,
    proposal_generator,
)
from blackjax.mcmc.trajectory import (
    Trajectory,
    append_to_trajectory,
    hmc_energy,
    merge_trajectories,
    reorder_trajectories,
)
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "NUTSInfo",
    "SoftAvgState",
    "init",
    "build_kernel",
    "as_top_level_api",
    "stop_gradient_blend",
    "update_soft_average",
    "merge_soft_averages",
]


init = hmc.init


class NUTSInfo(NamedTuple):
    """Additional information on the NUTS transition (same fields as BlackJAX)."""

    momentum: ArrayTree
    is_divergent: bool
    is_turning: bool
    energy: float
    trajectory_leftmost_state: integrators.IntegratorState
    trajectory_rightmost_state: integrators.IntegratorState
    num_trajectory_expansions: int
    num_integration_steps: int
    acceptance_rate: float


class SoftAvgState(NamedTuple):
    """Online softmax-weighted average of encountered integrator states.

    ``log_weight_sum`` is ``logsumexp`` of the per-state weights ``H0 - H``,
    so the implied mixture is ``softmax(-H)`` up to the constant ``H0``.
    """

    state: IntegratorState
    log_weight_sum: float


def update_soft_average(
    soft_avg: SoftAvgState,
    new_state: IntegratorState,
    new_weight: float,
) -> SoftAvgState:
    """Fold one proposal into the running softmax-weighted average."""
    log_weight_sum = jnp.logaddexp(soft_avg.log_weight_sum, new_weight)
    alpha = jnp.exp(new_weight - log_weight_sum)
    avg_state = jax.tree_util.tree_map(
        lambda a, s: (1.0 - alpha) * a + alpha * s,
        soft_avg.state,
        new_state,
    )
    return SoftAvgState(avg_state, log_weight_sum)


def merge_soft_averages(left: SoftAvgState, right: SoftAvgState) -> SoftAvgState:
    """Merge two online softmax averages (associative)."""
    log_weight_sum = jnp.logaddexp(left.log_weight_sum, right.log_weight_sum)
    alpha_right = jnp.exp(right.log_weight_sum - log_weight_sum)
    avg_state = jax.tree_util.tree_map(
        lambda a, b: (1.0 - alpha_right) * a + alpha_right * b,
        left.state,
        right.state,
    )
    return SoftAvgState(avg_state, log_weight_sum)


def stop_gradient_blend(soft_state, hard_state):
    """Forward: ``hard_state``; backward: gradients through ``soft_state``.

    Analogous to :func:`src.samplers.sigmoid_accept`:
    ``soft - stop_gradient(soft) + stop_gradient(hard)``.
    """
    return jax.tree_util.tree_map(
        lambda soft, hard: soft - jax.lax.stop_gradient(soft) + jax.lax.stop_gradient(hard),
        soft_state,
        hard_state,
    )


def empty_soft_average(placeholder: IntegratorState) -> SoftAvgState:
    """Soft average with no mass yet; first ``update_soft_average`` sets state fully."""
    return SoftAvgState(placeholder, -jnp.inf)


# -------------------------------------------------------------------
# Trajectory integration / expansion with online soft averaging
# -------------------------------------------------------------------


class DynamicIntegrationState(NamedTuple):
    step: int
    proposal: Proposal
    trajectory: Trajectory
    termination_state: NamedTuple
    soft_avg: SoftAvgState


class DynamicExpansionState(NamedTuple):
    step: int
    proposal: Proposal
    trajectory: Trajectory
    termination_state: NamedTuple
    soft_avg: SoftAvgState


def dynamic_progressive_integration(
    integrator: Callable,
    kinetic_energy: Callable,
    update_termination_state: Callable,
    is_criterion_met: Callable,
    divergence_threshold: float,
):
    """Like BlackJAX ``dynamic_progressive_integration``, also tracking soft averages."""
    _, generate_proposal = proposal_generator(hmc_energy(kinetic_energy))
    sample_proposal = progressive_uniform_sampling

    def integrate(
        rng_key: PRNGKey,
        initial_state: IntegratorState,
        direction: int,
        termination_state,
        max_num_steps: int,
        step_size,
        initial_energy,
    ):
        def do_keep_integrating(loop_state):
            integration_state, (is_diverging, has_terminated) = loop_state
            return (
                (integration_state.step < max_num_steps)
                & ~has_terminated
                & ~is_diverging
            )

        def add_one_state(loop_state):
            integration_state, _ = loop_state
            step, proposal_, traj, termination_state_, soft_avg = integration_state
            proposal_key = jax.random.fold_in(rng_key, step)

            new_state = integrator(traj.rightmost_state, direction * step_size)
            new_proposal = generate_proposal(initial_energy, new_state)
            is_diverging = -new_proposal.weight > divergence_threshold

            # Every newly integrated state is an encountered potential proposal.
            soft_avg = update_soft_average(soft_avg, new_state, new_proposal.weight)

            (new_trajectory, sampled_proposal) = jax.lax.cond(
                step == 0,
                lambda _: (
                    Trajectory(new_state, new_state, new_state.momentum, 1),
                    new_proposal,
                ),
                lambda _: (
                    append_to_trajectory(traj, new_state),
                    sample_proposal(proposal_key, proposal_, new_proposal),
                ),
                operand=None,
            )

            new_termination_state = update_termination_state(
                termination_state_,
                new_trajectory.momentum_sum,
                new_state.momentum,
                step,
            )
            has_terminated = is_criterion_met(
                new_termination_state, new_trajectory.momentum_sum, new_state.momentum
            )

            new_integration_state = DynamicIntegrationState(
                step + 1,
                sampled_proposal,
                new_trajectory,
                new_termination_state,
                soft_avg,
            )
            return (new_integration_state, (is_diverging, has_terminated))

        proposal_placeholder = generate_proposal(initial_energy, initial_state)
        trajectory_placeholder = Trajectory(
            initial_state, initial_state, initial_state.momentum, 0
        )
        integration_state_placeholder = DynamicIntegrationState(
            0,
            proposal_placeholder,
            trajectory_placeholder,
            termination_state,
            empty_soft_average(initial_state),
        )

        new_integration_state, (is_diverging, has_terminated) = jax.lax.while_loop(
            do_keep_integrating,
            add_one_state,
            (integration_state_placeholder, (False, False)),
        )
        _, proposal_, traj, termination_state_, soft_avg = new_integration_state

        new_trajectory = jax.lax.cond(
            direction > 0,
            lambda _: traj,
            lambda _: Trajectory(
                traj.rightmost_state,
                traj.leftmost_state,
                traj.momentum_sum,
                traj.num_states,
            ),
            operand=None,
        )

        return (
            proposal_,
            new_trajectory,
            termination_state_,
            is_diverging,
            has_terminated,
            soft_avg,
        )

    return integrate


def dynamic_multiplicative_expansion(
    trajectory_integrator: Callable,
    uturn_check_fn: Callable,
    max_num_expansions: int = 10,
    rate: int = 2,
) -> Callable:
    """Like BlackJAX ``dynamic_multiplicative_expansion``, merging soft averages."""
    proposal_sampler = progressive_biased_sampling

    def expand(
        rng_key: PRNGKey,
        initial_expansion_state: DynamicExpansionState,
        initial_energy: float,
        step_size: float,
    ):
        def do_keep_expanding(loop_state) -> bool:
            expansion_state, (is_diverging, is_turning) = loop_state
            return (
                (expansion_state.step < max_num_expansions)
                & ~is_diverging
                & ~is_turning
            )

        def expand_once(loop_state):
            expansion_state, _ = loop_state
            step, proposal_, traj, termination_state, soft_avg = expansion_state

            subkey = jax.random.fold_in(rng_key, step)
            direction_key, trajectory_key, proposal_key = jax.random.split(subkey, 3)

            direction = jnp.where(jax.random.bernoulli(direction_key), 1, -1)
            start_state = jax.lax.cond(
                direction > 0,
                lambda _: traj.rightmost_state,
                lambda _: traj.leftmost_state,
                operand=None,
            )
            (
                new_proposal,
                new_trajectory,
                termination_state,
                is_diverging,
                is_turning_subtree,
                subtree_soft_avg,
            ) = trajectory_integrator(
                trajectory_key,
                start_state,
                direction,
                termination_state,
                rate**step,
                step_size,
                initial_energy,
            )

            # Soft average always absorbs encountered subtree states, even when
            # the discrete proposal rejects a turning/diverging subtree.
            soft_avg = merge_soft_averages(soft_avg, subtree_soft_avg)

            def update_sum_log_p_accept(inputs):
                _, prop, new_prop = inputs
                return Proposal(
                    prop.state,
                    prop.energy,
                    prop.weight,
                    jnp.logaddexp(prop.sum_log_p_accept, new_prop.sum_log_p_accept),
                )

            updated_proposal = jax.lax.cond(
                is_diverging | is_turning_subtree,
                update_sum_log_p_accept,
                lambda x: proposal_sampler(*x),
                operand=(proposal_key, proposal_, new_proposal),
            )

            left_trajectory, right_trajectory = reorder_trajectories(
                direction, traj, new_trajectory
            )
            merged_trajectory = merge_trajectories(left_trajectory, right_trajectory)

            is_turning = uturn_check_fn(
                merged_trajectory.leftmost_state.momentum,
                merged_trajectory.rightmost_state.momentum,
                merged_trajectory.momentum_sum,
            )

            new_state = DynamicExpansionState(
                step + 1,
                updated_proposal,
                merged_trajectory,
                termination_state,
                soft_avg,
            )
            info = (is_diverging, is_turning_subtree | is_turning)
            return (new_state, info)

        expansion_state, (is_diverging, is_turning) = jax.lax.while_loop(
            do_keep_expanding,
            expand_once,
            (initial_expansion_state, (False, False)),
        )
        return expansion_state, (is_diverging, is_turning)

    return expand


def iterative_nuts_proposal(
    integrator: Callable,
    kinetic_energy: metrics.KineticEnergy,
    uturn_check_fn: metrics.CheckTurning,
    max_num_expansions: int = 10,
    divergence_threshold: float = 1000,
) -> Callable:
    """Iterative NUTS proposal with stop-gradient softmax averaging."""
    (
        new_termination_state,
        update_termination_state,
        is_criterion_met,
    ) = termination.iterative_uturn_numpyro(uturn_check_fn)

    trajectory_integrator = dynamic_progressive_integration(
        integrator,
        kinetic_energy,
        update_termination_state,
        is_criterion_met,
        divergence_threshold,
    )

    expand = dynamic_multiplicative_expansion(
        trajectory_integrator,
        uturn_check_fn,
        max_num_expansions,
    )

    def _compute_energy(state: IntegratorState) -> float:
        return -state.logdensity + kinetic_energy(state.momentum)

    def propose(rng_key, initial_state: IntegratorState, step_size):
        initial_termination_state = new_termination_state(
            initial_state, max_num_expansions
        )
        initial_energy = _compute_energy(initial_state)
        # Initial proposal weight is H0 - H0 = 0, matching BlackJAX.
        initial_proposal = Proposal(initial_state, initial_energy, 0.0, -np.inf)
        initial_trajectory = Trajectory(
            initial_state,
            initial_state,
            initial_state.momentum,
            0,
        )
        # Include the orbit's starting state in the soft average.
        initial_soft_avg = SoftAvgState(initial_state, 0.0)
        initial_expansion_state = DynamicExpansionState(
            0,
            initial_proposal,
            initial_trajectory,
            initial_termination_state,
            initial_soft_avg,
        )

        expansion_state, info = expand(
            rng_key, initial_expansion_state, initial_energy, step_size
        )
        is_diverging, is_turning = info
        num_doublings, sampled_proposal, new_trajectory, _, soft_avg = expansion_state

        acceptance_rate = (
            jnp.exp(sampled_proposal.sum_log_p_accept) / new_trajectory.num_states
        )

        # Discrete multinomial sample on the forward pass; softmax-weighted
        # average of all encountered proposals on the backward pass.
        blended_state = stop_gradient_blend(soft_avg.state, sampled_proposal.state)

        nuts_info = NUTSInfo(
            initial_state.momentum,
            is_diverging,
            is_turning,
            sampled_proposal.energy,
            new_trajectory.leftmost_state,
            new_trajectory.rightmost_state,
            num_doublings,
            new_trajectory.num_states,
            acceptance_rate,
        )
        return blended_state, nuts_info

    return propose


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: int = 1000,
):
    """Build an iterative NUTS kernel with stop-gradient softmax averaging."""

    def kernel(
        rng_key: PRNGKey,
        state: hmc.HMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
        max_num_doublings: int = 10,
    ) -> tuple[hmc.HMCState, NUTSInfo]:
        metric = metrics.default_metric(inverse_mass_matrix)
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        proposal_generator_fn = iterative_nuts_proposal(
            symplectic_integrator,
            metric.kinetic_energy,
            metric.check_turning,
            max_num_doublings,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad = state
        momentum = metric.sample_momentum(key_momentum, position)

        integrator_state = IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal_state, info = proposal_generator_fn(
            key_integrator, integrator_state, step_size
        )
        proposal_hmc = hmc.HMCState(
            proposal_state.position,
            proposal_state.logdensity,
            proposal_state.logdensity_grad,
        )
        return proposal_hmc, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: metrics.MetricTypes,
    *,
    max_num_doublings: int = 10,
    divergence_threshold: int = 1000,
    integrator: Callable = integrators.velocity_verlet,
) -> SamplingAlgorithm:
    """User-facing SamplingAlgorithm wrapping :func:`build_kernel`."""
    kernel = build_kernel(integrator, divergence_threshold)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            max_num_doublings,
        )

    return SamplingAlgorithm(init_fn, step_fn)
