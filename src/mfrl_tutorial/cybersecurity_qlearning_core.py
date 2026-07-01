# -*- coding: utf-8 -*-
"""Tabular Q-learning components for the cybersecurity MFC example."""

import os
import time
import itertools
from typing import Dict, List, Optional, Tuple, Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
import gymnasium as gym
from scipy.spatial import cKDTree

# Set seed for reproducibility
np.random.seed(1)

# Global Constants
N_STATES = 4  # Number of states: [DI, DS, UI, US]
v_H_DEFAULT = 0.6
ACTION_UPPER_BOUND = 1.0
ACTION_LOWER_BOUND = 0.0
SHOW_PLOTS = False

# Plotting Style
try:
    plt.style.use('seaborn-v0_8-muted')
except (ImportError, ValueError):
    try:
        plt.style.use('ggplot')
    except (ImportError, ValueError):
        pass

class CyberSecEnvContinuous(gym.Env):
    """
    Continuous population environment for Cyber Security Mean Field Control.
    Operates on population distributions (mu) without discretization.
    """
    def __init__(self, T: float, dt: float, cn_sampler: Callable, n_actions_level: int = 2):
        super(CyberSecEnvContinuous, self).__init__()
        self.T = T
        self.dt = dt
        self.Nt = int(self.T / self.dt)
        self.NS = N_STATES

        self.beta_UU, self.beta_UD = 0.3, 0.4
        self.beta_DU, self.beta_DD = 0.3, 0.4
        self.lambda_speed = 0.8
        self.q_rec_D, self.q_rec_U = 0.5, 0.4
        self.q_inf_D, self.q_inf_U = 0.4, 0.3
        self.k_D, self.k_I = 0.3, 0.5
        self.v_H = v_H_DEFAULT
        self.common_noise_sampler = cn_sampler

        self.mu = np.zeros(self.NS)
        self.t = 0.0

        # Discretize action space: n_actions_level per population state
        action_levels = np.linspace(ACTION_LOWER_BOUND, ACTION_UPPER_BOUND, n_actions_level)
        self.all_discrete_actions = np.array(list(itertools.product(action_levels, repeat=self.NS)))
        self.n_tabular_actions = len(self.all_discrete_actions)

    def get_dynamics_matrix(self, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        L = np.zeros((self.NS, self.NS))
        L[0, 1] = self.q_rec_D
        L[2, 3] = self.q_rec_U
        L[1, 0] = self.v_H * self.q_inf_D + self.beta_DD * mu[0] + self.beta_UD * mu[2]
        L[3, 2] = self.v_H * self.q_inf_U + self.beta_UU * mu[2] + self.beta_DU * mu[0]
        target_map = {0: 2, 1: 3, 2: 0, 3: 1}
        for i, j in target_map.items():
            L[i, j] += alpha[i] * self.lambda_speed
        for i in range(self.NS):
            L[i, i] = -np.sum(L[i, :])
        return L

    def get_reward(self, mu: np.ndarray, alpha: np.ndarray) -> float:
        costs_vec = np.zeros(self.NS)
        costs_vec[0] = self.k_D + self.k_I
        costs_vec[1] = self.k_D
        costs_vec[2] = self.k_I
        return -np.sum(mu * costs_vec) * self.dt

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        alpha = self.all_discrete_actions[action_idx]
        reward = self.get_reward(self.mu, alpha)
        L = self.get_dynamics_matrix(self.mu, alpha)
        self.mu = self.mu + self.dt * np.matmul(self.mu, L)
        self.mu = np.clip(self.mu, 1e-10, 1.0)
        self.mu /= np.sum(self.mu)
        self.t += self.dt
        done = (self.t > self.T - 0.5 * self.dt)
        self.v_H = self.common_noise_sampler(self.v_H)
        return self.mu, reward, done, False, {}

    def reset(self, pop_distrib: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        self.t = 0.0
        self.mu = pop_distrib.copy()
        self.v_H = v_H_DEFAULT
        return self.mu, {}

def sampler(n_states: int) -> np.ndarray:
    r = np.random.rand()
    if r < 0.4:
        alpha = [1.0] * n_states
        mu = np.random.dirichlet(alpha)
    elif r < 0.8:
        alpha = [0.1] * n_states
        mu = np.random.dirichlet(alpha)
    else:
        mu = np.zeros(n_states)
        n_active = np.random.randint(1, 4)
        indices = np.random.choice(n_states, n_active, replace=False)
        mu[indices] = np.random.dirichlet([1.0] * n_active)
    return mu

def plot_training_progress(output_dir: str, prefix: str, eval_rewards: List[float], eval_eps: List[int], ep: int, q_diffs: List[float], train_rewards: Optional[List[float]] = None, window_size: int = 20):
    # 1. Q-diff log plot
    if q_diffs:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(q_diffs, color='#348ABD', lw=1.0, alpha=0.3, label='Q-diff')
        if len(q_diffs) >= window_size:
            ma_q = np.convolve(q_diffs, np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size-1, len(q_diffs)), ma_q, color='#286082', lw=2.0, ls='--', label=f'Q-diff Moving Average ({window_size})')
        ax.set_yscale('log')
        ax.set_xlabel("Iterations", fontsize=14)
        ax.set_ylabel("Max Q-Change", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, ls='--', alpha=0.4, zorder=0)
        ax.legend(fontsize=12)
        plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_qdiff.pdf"), bbox_inches='tight')
        plt.close()

    # 2. Training and Evaluation Reward/Cost Plots
    if (train_rewards is not None and len(train_rewards) > 0) or (eval_rewards and eval_eps):
        has_train = train_rewards is not None and len(train_rewards) > 0
        has_eval = eval_rewards and eval_eps

        # Reward Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        if has_train:
            ax.plot(train_rewards, color='#E24A33', alpha=0.3, label='Train Reward')
            if len(train_rewards) >= window_size:
                ma = np.convolve(train_rewards, np.ones(window_size) / window_size, mode='valid')
                ax.plot(range(window_size-1, len(train_rewards)), ma, color='#E24A33', lw=2, ls='--', label=f'Train Reward Moving Avg ({window_size})')

        if has_eval:
            label = 'Eval Reward (Avg over Inits)' if has_train else 'Eval Reward'
            ax.plot(eval_eps, eval_rewards, color='#5E2D79', lw=2, marker='o' if not has_train else None, markersize=4, label=label)

        ax.set_xlabel("Iterations", fontsize=14)
        ax.set_ylabel("Total Reward" if has_train else "Average Reward", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, ls='--', alpha=0.6, zorder=0)
        ax.legend(loc='best', fontsize=12)
        plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_training.pdf"), bbox_inches='tight')
        plt.close()

        # Cost Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        if has_train:
            costs = [-r for r in train_rewards]
            ax.plot(costs, color='#348ABD', alpha=0.3, label='Train Cost')
            if len(costs) >= window_size:
                ma = np.convolve(costs, np.ones(window_size) / window_size, mode='valid')
                ax.plot(range(window_size-1, len(costs)), ma, color='#348ABD', lw=2, ls='--', label=f'Train Cost Moving Avg ({window_size})')

        if has_eval:
            eval_costs = [-r for r in eval_rewards]
            label = 'Eval Cost (Avg over Inits)' if has_train else 'Eval Cost'
            ax.plot(eval_eps, eval_costs, color='#1B5E20', lw=2, marker='o' if not has_train else None, markersize=4, label=label)

        ax.set_xlabel("Iterations", fontsize=14)
        ax.set_ylabel("Total Cost" if has_train else "Average Cost", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, ls='--', alpha=0.6, zorder=0)
        ax.legend(loc='best', fontsize=12)
        plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_training_cost.pdf"), bbox_inches='tight')
        plt.close()

def plot_evaluation(output_dir: str, prefix: str, T: float, Nt: int, trajs: List[List[np.ndarray]], ep: int, ode_trajs: Optional[List[np.ndarray]] = None):
    n_tests = len(trajs)
    if n_tests <= 0: return
    t_space = np.linspace(0, T, num=Nt + 1)
    labels, colors = ["DI", "DS", "UI", "US"], ['#E24A33', '#348ABD', '#8EBA42', '#988ED5']
    fig_test, axes = plt.subplots(1, n_tests, figsize=(4 * n_tests, 3.5), squeeze=False)
    for idx in range(n_tests):
        ax_t, data = axes[0, idx], np.asarray(trajs[idx])
        if ode_trajs is not None:
            ts = np.asarray(ode_trajs[idx])
            t_ode = np.linspace(0, T, num=ts.shape[0])
            for i in range(4): ax_t.plot(t_ode, ts[:, i], ls='--', lw=4.0, color=colors[i], alpha=0.3, zorder=5)
        for i in range(4):
            # Add a white "halo" outline to make the main line pop
            ax_t.plot(t_space, data[:, i], lw=3.0, color='white', alpha=0.7, zorder=9)
            # Main colored line
            ax_t.plot(t_space, data[:, i], label=labels[i] if idx == 0 else None, lw=1.5, ls='-', color=colors[i], alpha=1.0, zorder=10)

        ax_t.set_xlabel("Time")
        if idx == 0:
            ax_t.set_ylabel("Proportion")
            ax_t.legend()
        ax_t.set_ylim(0, 1.05)
        ax_t.grid(True, ls='--', alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_eval.pdf"), bbox_inches='tight')
    plt.close()


class ForwardBackwardSolver:
    def __init__(self, env: Any, rho: float, dt_fine: float = 0.01):
        self.T, self.dt = env.T, dt_fine
        self.Nt = int(self.T / self.dt)
        self.NS = N_STATES
        self.beta_UU, self.beta_UD = env.beta_UU, env.beta_UD
        self.beta_DU, self.beta_DD = env.beta_DU, env.beta_DD
        self.lambda_speed = env.lambda_speed
        self.q_rec_D, self.q_rec_U = env.q_rec_D, env.q_rec_U
        self.q_inf_D, self.q_inf_U = env.q_inf_D, env.q_inf_U
        self.k_D, self.k_I = env.k_D, env.k_I
        self.v_H = v_H_DEFAULT
        self.discount_beta = rho

    def get_index(self, s: str) -> int: return {'DI': 0, 'DS': 1, 'UI': 2, 'US': 3}.get(s)

    def get_lambda_t(self, mu_t: np.ndarray, u_t: np.ndarray, alpha_t: float) -> np.ndarray:
        lambda_matrix = np.zeros((self.NS, self.NS))
        lambda_matrix[self.get_index('DI'), self.get_index('DS')] = self.q_rec_D
        lambda_matrix[self.get_index('DS'), self.get_index('DI')] = (
            self.v_H * self.q_inf_D + self.beta_DD * mu_t[self.get_index('DI')] +
            self.beta_UD * mu_t[self.get_index('UI')]
        )
        lambda_matrix[self.get_index('UI'), self.get_index('US')] = self.q_rec_U
        lambda_matrix[self.get_index('US'), self.get_index('UI')] = (
            self.v_H * self.q_inf_U + self.beta_UU * mu_t[self.get_index('UI')] +
            self.beta_DU * mu_t[self.get_index('DI')]
        )
        if alpha_t > 0.5:
            lambda_matrix[self.get_index('DI'), self.get_index('UI')] = self.lambda_speed
            lambda_matrix[self.get_index('DS'), self.get_index('US')] = self.lambda_speed
            lambda_matrix[self.get_index('UI'), self.get_index('DI')] = self.lambda_speed
            lambda_matrix[self.get_index('US'), self.get_index('DS')] = self.lambda_speed
        for iS in range(self.NS): lambda_matrix[iS, iS] = -np.sum(lambda_matrix[iS])
        return lambda_matrix

    def solve(self, mu_0: np.ndarray, tol: float = 1e-5, max_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        mu, u = np.tile(mu_0, (self.Nt + 1, 1)), np.zeros((self.Nt + 1, self.NS))
        for i in range(max_iter):
            new_mu = self.solve_KFP(mu, u)
            new_u = self.solve_HJB(new_mu, u)
            if np.linalg.norm(new_mu - mu) < tol and np.linalg.norm(new_u - u) < tol: break
            mu, u = new_mu, new_u
        return mu, u

    def solve_KFP(self, mu: np.ndarray, u: np.ndarray) -> np.ndarray:
        new_mu = np.zeros((self.Nt + 1, self.NS))
        new_mu[0] = mu[0]
        for it in range(self.Nt):
            q_t = np.zeros((self.NS, self.NS))
            alphahat = self.get_alphahat_t_vec(mu[it], u[it])
            for iS in range(self.NS): q_t[iS] = self.get_lambda_t(mu[it], u[it], alphahat[iS])[iS]
            new_mu[it + 1] = np.matmul(new_mu[it], np.eye(self.NS) + self.dt * q_t)
        return new_mu

    def solve_HJB(self, mu: np.ndarray, u: np.ndarray) -> np.ndarray:
        new_u = np.zeros((self.Nt + 1, self.NS))
        for it in range(self.Nt - 1, -1, -1):
            opt_H, Dmu_opt_H = np.zeros(self.NS), np.zeros((self.NS, self.NS))
            for iS in range(self.NS):
                H0, H1 = self.get_Hamiltonian(iS, mu[it + 1], new_u[it + 1], 0), self.get_Hamiltonian(iS, mu[it + 1], new_u[it + 1], 1)
                alpha = 0 if H0 <= H1 else 1
                opt_H[iS] = H0 if alpha == 0 else H1
                for iSderiv in range(self.NS): Dmu_opt_H[iS][iSderiv] = self.get_Dmu_Hamiltonian(iS, iSderiv, mu[it + 1], new_u[it + 1], alpha)
            new_u[it] = new_u[it + 1] + self.dt * (opt_H + np.matmul(Dmu_opt_H, mu[it + 1]) - self.discount_beta * new_u[it + 1])
        return new_u

    def get_alphahat_t_vec(self, mu_t: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        alphahat = np.zeros(self.NS)
        for iS in range(self.NS):
            H0 = self.get_Hamiltonian(iS, mu_t, u_t, 0)
            H1 = self.get_Hamiltonian(iS, mu_t, u_t, 1)
            alphahat[iS] = 0 if H0 <= H1 else 1
        return alphahat

    def get_Hamiltonian(self, iS: int, mu_t: np.ndarray, u_t: np.ndarray, alpha_t: float) -> float:
        return np.matmul(self.get_lambda_t(mu_t, u_t, alpha_t)[iS], u_t) + self.running_cost_t(iS)

    def get_Dmu_Hamiltonian(self, iS: int, iSderiv: int, mu_t: np.ndarray, u_t: np.ndarray, alpha_t: float) -> float:
        return np.matmul(self.get_Dmu_lambda_t(iSderiv, mu_t, u_t, alpha_t)[iS], u_t)

    def get_Dmu_lambda_t(self, iSderiv: int, mu_t: np.ndarray, u_t: np.ndarray, alpha_t: float) -> np.ndarray:
        Dmu_lambda_matrix = np.zeros((self.NS, self.NS))
        if iSderiv == self.get_index('DI'):
            Dmu_lambda_matrix[self.get_index('DS'), self.get_index('DI')] = self.beta_DD
            Dmu_lambda_matrix[self.get_index('US'), self.get_index('UI')] = self.beta_DU
            Dmu_lambda_matrix[self.get_index('DS'), self.get_index('DS')] = -self.beta_DD
            Dmu_lambda_matrix[self.get_index('US'), self.get_index('US')] = -self.beta_DU
        if iSderiv == self.get_index('UI'):
            Dmu_lambda_matrix[self.get_index('DS'), self.get_index('DI')] = self.beta_UD
            Dmu_lambda_matrix[self.get_index('US'), self.get_index('UI')] = self.beta_UU
            Dmu_lambda_matrix[self.get_index('DS'), self.get_index('DS')] = -self.beta_UD
            Dmu_lambda_matrix[self.get_index('US'), self.get_index('US')] = -self.beta_UU
        return Dmu_lambda_matrix

    def running_cost_t(self, iS: int) -> float:
        rcost = 0.0
        if iS == self.get_index('DI') or iS == self.get_index('DS'): rcost += self.k_D
        if iS == self.get_index('DI') or iS == self.get_index('UI'): rcost += self.k_I
        return rcost

def precompute_tables(env: Any, discretization_params: Dict) -> Dict:
    # Faster Simplex Grid Generation: O(S) instead of O((n+1)^N_STATES)
    n_steps_state = discretization_params['n_steps_state']
    print(f"Generating simplex grid for n={n_steps_state}...")
    def generate_simplex_states(n, k):
        for c in itertools.combinations(range(n + k - 1), k - 1):
            yield [b - a - 1 for a, b in zip((-1,) + c, c + (n + k - 1,))]

    all_discrete_states = np.array(list(generate_simplex_states(n_steps_state, N_STATES)), dtype=float) / n_steps_state
    S = len(all_discrete_states)
    print(f"Generated {S} discrete states for n={n_steps_state}.")
    grid_tree = cKDTree(all_discrete_states)
    A = env.n_tabular_actions

    print(f"Precomputing {S}x{A} transitions/rewards...")
    mu_b = all_discrete_states[:, np.newaxis, :]
    al_b = env.all_discrete_actions[np.newaxis, :, :]
    L = np.zeros((S, A, 4, 4))
    L[:,:,0,1], L[:,:,2,3] = env.q_rec_D, env.q_rec_U
    L[:,:,1,0] = env.v_H*env.q_inf_D + env.beta_DD*mu_b[:,:,0] + env.beta_UD*mu_b[:,:,2]
    L[:,:,3,2] = env.v_H*env.q_inf_U + env.beta_UU*mu_b[:,:,2] + env.beta_DU*mu_b[:,:,0]
    for i, j in {0:2, 1:3, 2:0, 3:1}.items(): L[:,:,i,j] += al_b[:,:,i]*env.lambda_speed
    for i in range(4): L[:,:,i,i] = -np.sum(L[:,:,i,:], axis=2)

    next_mu_c = mu_b + env.dt * np.einsum('saj,sajk->sak', mu_b, L)
    next_mu_c = np.clip(next_mu_c, 1e-10, 1.0)
    next_mu_c /= np.sum(next_mu_c, axis=2, keepdims=True)

    costs = np.array([env.k_D+env.k_I, env.k_D, env.k_I, 0.0])
    # mu_b is (S, 1, 4), al_b is (1, A, 4), but rewards only depend on mu and action is fixed per level.
    # However, env.get_reward(mu, alpha) in CyberSecEnvContinuous actually doesn't use alpha!
    # Let's check env.get_reward:
    # def get_reward(self, mu: np.ndarray, alpha: np.ndarray) -> float:
    #     costs_vec = np.zeros(self.NS)
    #     costs_vec[0] = self.k_D + self.k_I
    #     costs_vec[1] = self.k_D
    #     costs_vec[2] = self.k_I
    #     return -np.sum(mu * costs_vec) * self.dt
    # It indeed only depends on mu. So reward_table is actually (S, 1) conceptually but we need it (S, A) for indexing.
    reward_table = -np.sum(mu_b * costs, axis=2) * env.dt # Shape (S, 1)
    reward_table = np.tile(reward_table, (1, A)) # Shape (S, A)

    print(f"Projecting {S*A} next states to grid...")
    _, transition_table = grid_tree.query(next_mu_c.reshape(-1, 4), p=1)
    transition_table = transition_table.reshape(S, A)

    print(f"Precomputation Complete | Reward Table: {reward_table.shape} | Transition Table: {transition_table.shape}")

    return {
        'states': all_discrete_states,
        'grid_tree': grid_tree,
        'reward_table': reward_table,
        'transition_table': transition_table,
        'S': S,
        'A': A
    }

def precompute_bench_values(solver: 'ForwardBackwardSolver', mu_DI_vals: List[float], grid_res: int) -> List[np.ndarray]:
    """Pre-calculates Benchmark Value Function V(m) = <u(0), m> for simplex slices."""
    print(f"Pre-computing {len(mu_DI_vals)} Benchmark Simplex Slices (res={grid_res})...")
    start_t = time.time()
    x = np.linspace(0, 1.0, grid_res)
    y = np.linspace(0, 1.0, grid_res)
    X, Y = np.meshgrid(x, y)
    bench_slices = []

    for di in mu_DI_vals:
        V_bench = np.full((grid_res, grid_res), np.nan)
        for ix in range(grid_res):
            for iy in range(grid_res):
                ds, ui = X[ix, iy], Y[ix, iy]
                if di + ds + ui <= 1.0001:
                    us = max(0, 1.0 - (di + ds + ui))
                    m = np.array([di, ds, ui, us])
                    _, u_traj = solver.solve(m, max_iter=20)
                    V_bench[ix, iy] = np.dot(u_traj[0], m)
        bench_slices.append(V_bench)
    print(f"Benchmark Pre-computation Complete | Time: {time.time()-start_t:.2f}s")
    return bench_slices

def plot_value_simplex_slices(output_dir: str, prefix: str, iter_idx: int, Q: np.ndarray, tables: Dict, grid_res: int = 10):
    """
    Visualizes the Value Function V(m) (Cost convention) on slices of the population simplex.
    Row 1: RL Cost (-max_a Q(m, a))
    Row 2: Benchmark Cost (<u(0), m>)
    Harmonized color scale and single colorbar.
    """
    mu_DI_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_slices = len(mu_DI_vals)

    # Grid for DS and UI
    x = np.linspace(0, 1.0, grid_res)
    y = np.linspace(0, 1.0, grid_res)
    X, Y = np.meshgrid(x, y)

    rl_costs_all = []
    bench_costs_all = []

    for j, di in enumerate(mu_DI_vals):
        V_rl = np.full((grid_res, grid_res), np.nan)
        V_bench = tables['bench_slices'][j]

        for ix in range(grid_res):
            for iy in range(grid_res):
                ds, ui = X[ix, iy], Y[ix, iy]
                if di + ds + ui <= 1.0001:
                    us = max(0, 1.0 - (di + ds + ui))
                    m = np.array([di, ds, ui, us])

                    # RL Cost = -max_a Q(m, a)
                    s_idx = tables['grid_tree'].query(m[np.newaxis, :], p=1)[1][0]
                    V_rl[ix, iy] = -np.max(Q[s_idx])

        rl_costs_all.append(V_rl)
        bench_costs_all.append(V_bench)

    # Harmonize color range across all slices and both rows
    all_vals = np.concatenate([np.array(rl_costs_all).flatten(), np.array(bench_costs_all).flatten()])
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)

    # Figure 1: RL Cost Slices
    fig_rl, axes_rl = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5), constrained_layout=True)
    if n_slices == 1: axes_rl = [axes_rl]
    for j, di in enumerate(mu_DI_vals):
        im = axes_rl[j].pcolormesh(X, Y, rl_costs_all[j], cmap='magma', shading='auto', vmin=vmin, vmax=vmax)
        axes_rl[j].set_title(f"RL Cost | $\mu_{{DI}}$={di}", fontsize=12)
        axes_rl[j].set_xlabel("$\mu_{DS}$")
        axes_rl[j].set_ylabel("$\mu_{UI}$")
        axes_rl[j].set_aspect('equal')
        fig_rl.colorbar(im, ax=axes_rl[j], shrink=0.8)
    plt.savefig(os.path.join(output_dir, f"{prefix}ep{iter_idx}_value_simplex_RL.pdf"), bbox_inches='tight')
    plt.close(fig_rl)

    # Figure 2: Benchmark Cost Slices
    fig_bench, axes_bench = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5), constrained_layout=True)
    if n_slices == 1: axes_bench = [axes_bench]
    for j, di in enumerate(mu_DI_vals):
        im = axes_bench[j].pcolormesh(X, Y, bench_costs_all[j], cmap='magma', shading='auto', vmin=vmin, vmax=vmax)
        axes_bench[j].set_title(f"Benchmark Cost | $\mu_{{DI}}$={di}", fontsize=12)
        axes_bench[j].set_xlabel("$\mu_{DS}$")
        axes_bench[j].set_ylabel("$\mu_{UI}$")
        axes_bench[j].set_aspect('equal')
        fig_bench.colorbar(im, ax=axes_bench[j], shrink=0.8)
    plt.savefig(os.path.join(output_dir, f"{prefix}ep{iter_idx}_value_simplex_Bench.pdf"), bbox_inches='tight')
    plt.close(fig_bench)

class QlearningSolver:
    def __init__(self, name: str, env: Any, tables: Dict, hyperparams: Dict, base_output_dir: str):
        self.name = name
        self.env = env
        self.tables = tables
        self.hyperparams = hyperparams
        self.Q = np.zeros((tables['S'], tables['A']))

        # Setup sub-directory within the base output directory
        self.output_dir = os.path.join(base_output_dir, name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_file = os.path.join(self.output_dir, "q_training.log")
        with open(self.log_file, "w") as f:
            f.write(f"Q-learning log | Name: {name} | Mode: {hyperparams['mode']}\n")
            f.write(f"Iter, Q-Diff, Epsilon, LR, Time\n")

    def project(self, mu):
        if mu.ndim == 1: mu = mu[np.newaxis, :]
        _, idx = self.tables['grid_tree'].query(mu, p=1)
        return idx

    def train(self, ode_sol: ForwardBackwardSolver, ode_trajs: Optional[List[np.ndarray]] = None):
        hp = self.hyperparams
        mode = hp['mode']
        total_iters = hp['total_iters']
        discount = hp['discount']
        lr_start, lr_end = hp['lr_start'], hp['lr_end']
        epsilon, epsilon_end, epsilon_decay = hp['epsilon'], hp['epsilon_end'], hp['epsilon_decay']
        eval_freq, plot_freq = hp['eval_freq'], hp['plot_freq']

        S, A = self.tables['S'], self.tables['A']
        reward_table = self.tables['reward_table']
        transition_table = self.tables['transition_table']
        state_range = np.arange(S)

        q_diffs, eval_rewards, eval_eps, train_rewards = [], [], [], []
        eval_inits = [np.array([0.25]*4), np.array([1.0, 0, 0, 0]), np.array([0, 0, 0, 1.0]), np.array([0.5, 0.2, 0.2, 0.1])]

        print(f"Starting Training: {self.name} ({mode}) | States: {S} | Actions: {A} | Iters: {total_iters}")

        for it in range(total_iters):
            t0 = time.time()
            cur_lr = max(lr_end, lr_start - (lr_start - lr_end) * (it / total_iters))
            max_delta = 0.0

            if mode == 'sync_complete':
                V_next = np.max(self.Q, axis=1)[transition_table]
                td_target = reward_table + discount * V_next
                delta = td_target - self.Q
                self.Q += cur_lr * delta
                max_delta = np.max(np.abs(delta))

            elif mode == 'sync_greedy':
                best_a = np.argmax(self.Q, axis=1)
                rand_a = np.random.randint(A, size=S)
                use_rand = np.random.rand(S) < epsilon
                a_idx = np.where(use_rand, rand_a, best_a)

                n_next = transition_table[state_range, a_idx]
                r = reward_table[state_range, a_idx]
                v_n = np.max(self.Q, axis=1)[n_next]
                delta = (r + discount * v_n) - self.Q[state_range, a_idx]
                self.Q[state_range, a_idx] += cur_lr * delta
                max_delta = np.max(np.abs(delta))

            elif mode == 'async_trajectory':
                mu, _ = self.env.reset(sampler(N_STATES))
                done = False
                tot_r = 0.0
                while not done:
                    s_idx = self.project(mu)[0]
                    a_idx = np.random.randint(A) if np.random.rand() < epsilon else np.argmax(self.Q[s_idx])

                    n_idx = transition_table[s_idx, a_idx]
                    r = reward_table[s_idx, a_idx]
                    delta = (r + discount * np.max(self.Q[n_idx])) - self.Q[s_idx, a_idx]
                    self.Q[s_idx, a_idx] += cur_lr * delta
                    max_delta = max(max_delta, abs(delta))
                    mu, rs_step, done, _, _ = self.env.step(a_idx)
                    tot_r += rs_step
                train_rewards.append(tot_r)

            q_diffs.append(max_delta)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if it == 0 or (it + 1) % 1000 == 0:
                elapsed = time.time()-t0
                print(f"[{self.name}] Iter {it+1}/{total_iters} | Diff: {max_delta:.4e} | Eps: {epsilon:.4f} | LR: {cur_lr:.3f} | Time: {elapsed:.4f}s")
                with open(self.log_file, "a") as f:
                    f.write(f"{it+1}, {max_delta:.4e}, {epsilon:.4f}, {cur_lr:.3f}, {elapsed:.4f}\n")

            if it == 0 or (it + 1) % eval_freq == 0:
                total_r = 0
                all_trajs = []
                for m0 in eval_inits:
                    mu_e, _ = self.env.reset(m0)
                    tr, d, traj = 0.0, False, [mu_e.copy()]
                    while not d:
                        mu_e, rc, d, _, _ = self.env.step(np.argmax(self.Q[self.project(mu_e)[0]]))
                        tr += rc
                        traj.append(mu_e.copy())
                    total_r += tr
                    all_trajs.append(traj)
                eval_rewards.append(total_r / len(eval_inits))
                eval_eps.append(it + 1)

                if it == 0 or (it + 1) % plot_freq == 0:
                    plot_training_progress(self.output_dir, "qlearning_", eval_rewards, eval_eps, it+1, q_diffs,
                                           train_rewards if mode == 'async_trajectory' else None)
                    plot_evaluation(self.output_dir, "qlearning_", self.env.T, self.env.Nt, all_trajs, it+1, ode_trajs)
                    plot_value_simplex_slices(self.output_dir, "qlearning_", it+1, self.Q, self.tables)

        np.save(os.path.join(self.output_dir, "Q_table.npy"), self.Q)
        print(f"Done with {self.name}! Results in {self.output_dir}")
