# -*- coding: utf-8 -*-
"""DDPG components for the cybersecurity MFC example."""

import os
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from itertools import combinations, product

import gymnasium as gym
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # Enable headless plotting
import numpy as np
import scipy as sp
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

# Set seed for reproducibility
np.random.seed(1)
torch.manual_seed(1)

# Global Constants
N_STATES = 4  # Number of states: [DI, DS, UI, US]
v_H_DEFAULT = 0.6
ACTION_UPPER_BOUND = 1.0
ACTION_LOWER_BOUND = 0.0
SHOW_PLOTS = False  # Toggle to True to display plots interactively

try:
    plt.style.use('seaborn-v0_8-muted')
except (ImportError, ValueError):
    try:
        plt.style.use('ggplot')
    except (ImportError, ValueError):
        pass


class CyberSecEnv(gym.Env):
    """
    Custom Environment that follows gym interface for a cyber security model.
    """

    def __init__(self, pop_distrib: np.ndarray, T: float, dt: float, cn_sampler: Callable):
        """
        Initializes the Cybersecurity environment.

        Args:
            pop_distrib (np.ndarray): Initial population distribution over states.
            T (float): Total time horizon.
            dt (float): Time step size.
            cn_sampler (Callable): Function to sample common noise (v_H).
        """
        super(CyberSecEnv, self).__init__()
        self.T = T
        self.dt = dt
        self.Nt = int(self.T / self.dt)
        self.NS = N_STATES

        self.beta_UU = 0.3
        self.beta_UD = 0.4
        self.beta_DU = 0.3
        self.beta_DD = 0.4
        self.lambda_speed = 0.8
        self.q_rec_D = 0.5
        self.q_rec_U = 0.4
        self.q_inf_D = 0.4
        self.q_inf_U = 0.3
        self.k_D = 0.3
        self.k_I = 0.5
        self.common_noise_sampler = cn_sampler
        self.v_H = v_H_DEFAULT

        self.action_space = spaces.Box(
            low=ACTION_LOWER_BOUND, high=ACTION_UPPER_BOUND,
            dtype=np.float64, shape=(self.NS,)
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, dtype=np.float64, shape=(self.NS,)
        )

        self.pop_distrib_discrete_init = pop_distrib.copy()
        self.current_proc_iS = 0
        self.reset(self.pop_distrib_discrete_init)

    def get_index(self, state_str: str) -> int:
        """
        Maps a state string name to its corresponding integer index.

        Args:
            state_str (str): One of ['DI', 'DS', 'UI', 'US'].

        Returns:
            int: The index of the state.
        """
        indices = {'DI': 0, 'DS': 1, 'UI': 2, 'US': 3}
        return indices.get(state_str)

    def running_cost_t(self, iS: int, mu: np.ndarray) -> float:
        """
        Calculates the running cost for a single agent in state iS given the mean field mu.

        Args:
            iS (int): State index.
            mu (np.ndarray): The current mean field (population distribution).

        Returns:
            float: The calculated cost.
        """
        rcost = 0.0
        if iS == self.get_index('DI') or iS == self.get_index('DS'):
            rcost += self.k_D
        if iS == self.get_index('DI') or iS == self.get_index('UI'):
            rcost += self.k_I
        return rcost

    def get_lambda_t_continuousAlpha(self, mu_t: np.ndarray, alpha_t: float) -> np.ndarray:
        """
        Computes the transition rate matrix (lambda) given the mean field and action.

        Args:
            mu_t (np.ndarray): Current population distribution.
            alpha_t (float): Controlled transition rate (action).

        Returns:
            np.ndarray: Matrix of transition rates between states.
        """
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
        target_states = {0: 2, 1: 3, 2: 0, 3: 1}  # DI<->UI, DS<->US
        target = target_states[self.current_proc_iS]
        lambda_matrix[self.current_proc_iS, target] = alpha_t * self.lambda_speed

        for iS in range(self.NS):
            lambda_matrix[iS, iS] = -np.sum(lambda_matrix[iS])
        return lambda_matrix

    def get_mu_and_reward(self, mu: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predicts the next mean field and calculates the total social reward.

        Args:
            mu (np.ndarray): Current population distribution.
            alpha (np.ndarray): Actions (one rate per state).

        Returns:
            Tuple[np.ndarray, float]: Next population distribution and immediate reward.
        """
        social_reward = 0.0
        new_mu_prev = mu
        q_t = np.zeros((self.NS, self.NS))
        for iS in range(self.NS):
            self.current_proc_iS = iS
            q_t[iS] = self.get_lambda_t_continuousAlpha(new_mu_prev, alpha[iS])[iS]

        new_mu = np.matmul(new_mu_prev, np.eye(self.NS) + self.dt * q_t)
        running_cost_vec = np.array([self.running_cost_t(i, new_mu_prev) for i in range(self.NS)])
        social_reward += np.inner(new_mu, -running_cost_vec) * self.dt
        return new_mu, social_reward

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Handle formatting differences
        if isinstance(action, list):
            action = action[0]

        mu = self.state
        new_mu, social_reward = self.get_mu_and_reward(mu, action)
        self.state = new_mu
        self.t += self.dt
        done = (self.t > self.T - 0.5 * self.dt)
        self.v_H = self.common_noise_sampler(self.v_H)
        return self.state, social_reward, done, False, {}

    def reset(self, pop_distrib: Optional[np.ndarray] = None, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.t = 0.0
        self.v_H = v_H_DEFAULT
        if pop_distrib is not None:
            self.state = pop_distrib
        else:
            self.state = self.pop_distrib_discrete_init
        return self.state, {}


class ForwardBackwardSolver:
    """
    Continuous-time forward-backward ODE solver for the Mean Field Control benchmark.
    """
    def __init__(self, env: Any, rho: float, dt_fine: float = 0.01):
        """
        Initializes the solver with parameters from the environment.

        Args:
            env (Any): The environment object to copy parameters from.
            rho (float): Continuous discount rate.
            dt_fine (float): Time step for the benchmark solver internal grid.
        """
        self.T = env.T
        self.dt = dt_fine
        self.Nt = int(self.T / self.dt)
        self.NS = N_STATES
        self.beta_UU = env.beta_UU
        self.beta_UD = env.beta_UD
        self.beta_DU = env.beta_DU
        self.beta_DD = env.beta_DD
        self.lambda_speed = env.lambda_speed
        self.q_rec_D = env.q_rec_D
        self.q_rec_U = env.q_rec_U
        self.q_inf_D = env.q_inf_D
        self.q_inf_U = env.q_inf_U
        self.k_D = env.k_D
        self.k_I = env.k_I
        self.v_H = v_H_DEFAULT
        self.discount_beta = rho

    def get_index(self, state_str: str) -> int:
        indices = {'DI': 0, 'DS': 1, 'UI': 2, 'US': 3}
        return indices.get(state_str)

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

        for iS in range(self.NS):
            lambda_matrix[iS, iS] = -np.sum(lambda_matrix[iS])
        return lambda_matrix

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
        if iS == self.get_index('DI') or iS == self.get_index('DS'):
            rcost += self.k_D
        if iS == self.get_index('DI') or iS == self.get_index('UI'):
            rcost += self.k_I
        return rcost

    def get_Hamiltonian(self, iS: int, mu_t: np.ndarray, u_t: np.ndarray, alpha_t: float) -> float:
        return np.matmul(self.get_lambda_t(mu_t, u_t, alpha_t)[iS], u_t) + self.running_cost_t(iS)

    def get_Dmu_Hamiltonian(self, iS: int, iSderiv: int, mu_t: np.ndarray, u_t: np.ndarray, alpha_t: float) -> float:
        return np.matmul(self.get_Dmu_lambda_t(iSderiv, mu_t, u_t, alpha_t)[iS], u_t)

    def get_alphahat_t_vec(self, mu_t: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        alphahat = np.zeros(self.NS)
        for iS in range(self.NS):
            H0 = self.get_Hamiltonian(iS, mu_t, u_t, 0)
            H1 = self.get_Hamiltonian(iS, mu_t, u_t, 1)
            alphahat[iS] = 0 if H0 <= H1 else 1
        return alphahat

    def solve_KFP(self, mu: np.ndarray, u: np.ndarray) -> np.ndarray:
        new_mu = np.zeros((self.Nt + 1, self.NS))
        new_mu[0] = mu[0]
        for it in range(self.Nt):
            q_t = np.zeros((self.NS, self.NS))
            alphahat = self.get_alphahat_t_vec(mu[it], u[it])
            for iS in range(self.NS):
                q_t[iS] = self.get_lambda_t(mu[it], u[it], alphahat[iS])[iS]
            new_mu[it + 1] = np.matmul(new_mu[it], np.eye(self.NS) + self.dt * q_t)
        return new_mu

    def solve_HJB(self, mu: np.ndarray, u: np.ndarray) -> np.ndarray:
        new_u = np.zeros((self.Nt + 1, self.NS))
        for it in range(self.Nt - 1, -1, -1):
            opt_H = np.zeros(self.NS)
            Dmu_opt_H = np.zeros((self.NS, self.NS))
            for iS in range(self.NS):
                H0 = self.get_Hamiltonian(iS, mu[it + 1], new_u[it + 1], 0)
                H1 = self.get_Hamiltonian(iS, mu[it + 1], new_u[it + 1], 1)
                alpha = 0 if H0 <= H1 else 1
                opt_H[iS] = H0 if alpha == 0 else H1
                for iSderiv in range(self.NS):
                    Dmu_opt_H[iS][iSderiv] = self.get_Dmu_Hamiltonian(iS, iSderiv, mu[it + 1], new_u[it + 1], alpha)
            new_u[it] = new_u[it + 1] + self.dt * opt_H + self.dt * np.matmul(Dmu_opt_H, mu[it + 1]) \
                        - self.dt * self.discount_beta * new_u[it + 1]
        return new_u

    def solve(self, mu_start: np.ndarray, tol: float = 1e-5, max_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iteratively solves the KFP-HJB system until convergence.

        Args:
            mu_start (np.ndarray): Initial state distribution at t=0.
            tol (float): Convergence tolerance.
            max_iter (int): Maximum number of forward-backward iterations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Converged population distribution (mu) and value function (u) trajectories.
        """
        mu = np.tile(mu_start, (self.Nt + 1, 1))
        u = np.zeros((self.Nt + 1, self.NS))
        for i in range(max_iter):
            new_mu = self.solve_KFP(mu, u)
            new_u = self.solve_HJB(new_mu, u)
            diff = np.linalg.norm(new_mu - mu) + np.linalg.norm(new_u - u)
            if diff < tol:
                # print(f"  ODE Converged at iter {i+1}")
                break
            mu, u = new_mu, new_u
        return mu, u


class ReplayBuffer:
    """Experience replay buffer for DDPG, storing PyTorch tensors."""
    def __init__(self, action_shape: Tuple, num_states: int, device: torch.device, capacity: int = 100000, batch_size: int = 64):
        """
        Initializes the buffer with pre-allocated tensors.

        Args:
            action_shape (Tuple): Shape of the action space.
            num_states (int): Number of states in the mean field.
            device (torch.device): Device where tensors reside (CPU/GPU).
            capacity (int): Max number of transitions to store.
            batch_size (int): Size of batches returned by sample().
        """
        self.capacity, self.batch_size, self.counter = capacity, batch_size, 0
        self.device = device
        self.s_buf = torch.zeros((capacity, num_states), dtype=torch.float32, device=device)
        self.a_buf = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=device)
        self.r_buf = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.ns_buf = torch.zeros((capacity, num_states), dtype=torch.float32, device=device)

    def record(self, obs: Tuple):
        """
        Stores a transition in the buffer.

        Args:
            obs (Tuple): (state, action, reward, next_state).
        """
        idx = self.counter % self.capacity
        state, action, reward, next_state = obs
        self.s_buf[idx] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.a_buf[idx] = torch.tensor(action[0] if isinstance(action, (list, tuple)) else action, dtype=torch.float32, device=self.device)
        self.r_buf[idx] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.ns_buf[idx] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.counter += 1

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random batch of transitions directly from the device.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: (s, a, r, ns) batches.
        """
        idx = np.random.randint(0, min(self.counter, self.capacity), size=self.batch_size)
        return (self.s_buf[idx], self.a_buf[idx], self.r_buf[idx], self.ns_buf[idx])


class Actor(nn.Module):
    """
    Actor network for MFC Policy.
    """
    def __init__(self, ns: int, na: int, hidden_dims: List[int] = [100, 100], activation: Any = nn.Sigmoid):
        """
        Initializes the Actor network with configurable layers.

        Args:
            ns (int): Number of states.
            na (int): Number of actions.
            hidden_dims (List[int]): List of hidden layer dimensions.
            activation (Any): Activation function class to use between layers.
        """
        super(Actor, self).__init__()
        layers = []
        last_dim = ns
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(activation())
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, na))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

        # Initialize final layer weights to be small as per DDPG literature
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Actor.
        """
        return self.net(x) * ACTION_UPPER_BOUND


class Critic(nn.Module):
    """
    Critic network for MFC Value estimation.
    """
    def __init__(self, ns: int, na_total: int, hidden_dims: List[int] = [100, 100], activation: Any = nn.Sigmoid):
        """
        Initializes the Critic network with configurable layers.

        Args:
            ns (int): Number of states.
            na_total (int): Total action dimension.
            hidden_dims (List[int]): List of hidden layer dimensions.
            activation (Any): Activation function class to use between layers.
        """
        super(Critic, self).__init__()
        layers = []
        last_dim = ns + na_total
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(activation())
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Critic. Concatenates state and action first.
        """
        a_flat = a.view(a.size(0), -1)
        x = torch.cat([s, a_flat], -1)
        return self.net(x)


def update_target(target: nn.Module, source: nn.Module, tau: float):
    """
    Performs a soft update of target network weights.

    Args:
        target (nn.Module): Target network to update.
        source (nn.Module): Main network with fresh weights.
        tau (float): Interpolation factor (0 < tau << 1).
    """
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(sp.data * tau + tp.data * (1 - tau))


# OUActionNoise class removed (switching to Gaussian)


def get_policy_action(state: np.ndarray, actor: nn.Module, noise: np.ndarray, device: torch.device) -> List[np.ndarray]:
    """
    Selects an action from the actor policy and adds Gaussian exploration noise.

    Args:
        state (np.ndarray): Current population distribution.
        actor (nn.Module): Policy network.
        noise (np.ndarray): Gaussian noise array.
        device (torch.device): Computation device.

    Returns:
        List[np.ndarray]: List containing the selected action array.
    """
    st = torch.FloatTensor(state).unsqueeze(0).to(device)
    actor.eval()
    with torch.no_grad():
        a = actor(st).cpu().numpy().squeeze(0)
    actor.train()
    return [np.clip(a + noise, 0, ACTION_UPPER_BOUND)]


def plot_training_progress(output_dir: str, prefix: str, rewards: List[float], ep: int, eval_rewards: Optional[List[float]] = None, eval_eps: Optional[List[int]] = None, window_size: int = 20):
    """
    Generates and saves plots for episode rewards and costs over training.

    Args:
        output_dir (str): Directory to save the plots.
        prefix (str): Filename prefix.
        rewards (List[float]): List of total rewards per episode.
        ep (int): Current episode index (for title).
        eval_rewards (Optional[List[float]]): List of average evaluation rewards.
        eval_eps (Optional[List[int]]): Episode indices where evaluation was performed.
        window_size (int): Window size for moving average calculation.
    """
    # Reward Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rewards, color='#E24A33', alpha=0.3, label='Train Reward')
    if len(rewards) >= window_size:
        ma = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        ax.plot(range(window_size - 1, len(rewards)), ma, color='#E24A33', lw=2, ls='--', label=f'Train Reward Moving Avg ({window_size})')

    if eval_rewards and eval_eps:
        ax.plot(eval_eps, eval_rewards, color='#5E2D79', lw=2, label='Eval Reward (Avg over Inits)')

    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Total Reward", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, ls='--', alpha=0.6)
    ax.legend(loc='best', fontsize=12)
    plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_training.pdf"), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Cost Plot
    costs = [-r for r in rewards]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(costs, color='#348ABD', alpha=0.3, label='Train Cost')
    if len(costs) >= window_size:
        ma = np.convolve(costs, np.ones(window_size) / window_size, mode='valid')
        ax.plot(range(window_size - 1, len(costs)), ma, color='#348ABD', lw=2, ls='--', label=f'Train Cost Moving Avg ({window_size})')

    if eval_rewards and eval_eps:
        eval_costs = [-r for r in eval_rewards]
        ax.plot(eval_eps, eval_costs, color='#1B5E20', lw=2, label='Eval Cost (Avg over Inits)')

    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Total Cost", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, ls='--', alpha=0.6)
    ax.legend(loc='best', fontsize=12)
    plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_training_cost.pdf"), bbox_inches='tight')
    plt.close()


def plot_evaluation(output_dir: str, prefix: str, env: Any, trajs: List[List[np.ndarray]], ep: int, true_sols: Optional[List[np.ndarray]] = None, actor_model: Optional[nn.Module] = None, device: Optional[torch.device] = None):
    """
    Generates evaluation plots comparing RL trajectories with baseline ODE solutions.

    Args:
        output_dir (str): Directory to save the plots.
        prefix (str): Filename prefix.
        env (Any): Simulation environment.
        trajs (List[List[np.ndarray]]): List of trajectories (distributions over time).
        ep (int): Current episode.
        true_sols (Optional[List[np.ndarray]]): Baseline solution for comparison.
        actor_model (Optional[nn.Module]): Policy model to visualize action distributions.
        device (Optional[torch.device]): Device for actor inference.
    """
    n_tests = len(trajs) - 1
    if n_tests <= 0: return

    plt.close('all')
    t_space = np.linspace(0, env.T, num=env.Nt + 1)
    labels, colors = ["DI", "DS", "UI", "US"], ['#E24A33', '#348ABD', '#8EBA42', '#988ED5']

    fig_test, axes = plt.subplots(1, n_tests, figsize=(4 * n_tests, 3.5), squeeze=False)
    for idx in range(n_tests):
        ax_t, data = axes[0, idx], np.asarray(trajs[idx + 1])

        # Plot benchmark (dashed) first with larger width
        if true_sols is not None:
            ts = np.asarray(true_sols[idx])
            t_ode = np.linspace(0, env.T, num=ts.shape[0])
            for i in range(4):
                ax_t.plot(t_ode, ts[:, i], ls='--', lw=4.0, color=colors[i], alpha=0.3, zorder=5)

        # Plot RL results (full) on top
        for i in range(4):
            # Add a white "halo" outline to make the main line pop
            ax_t.plot(t_space, data[:, i], lw=3.0, color='white', alpha=0.7, zorder=9)
            # Main colored line
            ax_t.plot(t_space, data[:, i], label=labels[i] if idx == 0 else None, lw=1.5, ls='-', color=colors[i], alpha=1.0, zorder=10)

        ax_t.set_xlabel("Time", fontsize=14)
        if idx == 0:
            ax_t.set_ylabel("Proportion", fontsize=14)
            ax_t.legend(loc='best', fontsize=12)
        ax_t.set_ylim(0, 1.05)
        ax_t.tick_params(axis='both', which='major', labelsize=12)
        ax_t.grid(True, ls='--', alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_eval.pdf"), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    if actor_model is not None and device is not None and true_sols is not None:
        for idx in range(n_tests):
            plt.close('all')
            x_space = np.arange(1, env.NS + 1)
            pop_list = trajs[idx + 1]
            pop_init = np.asarray(pop_list[0])
            pop_last = np.asarray(pop_list[-1])
            pop_window = 20
            pop_avg = np.mean(pop_list[-pop_window:], axis=0) if len(pop_list) >= pop_window else pop_last

            target_distribution = np.asarray(true_sols[idx][-1])

            fig, axs = plt.subplots(1, 3, figsize=(15, 4))

            # State Distribution
            axs[0].bar(x_space - 0.2, pop_init, width=0.2, label="init", color='green', alpha=0.7)
            axs[0].bar(x_space, pop_last, width=0.2, label="last", color='purple', alpha=0.7)
            axs[0].bar(x_space + 0.2, pop_avg, width=0.2, label=f"avg{pop_window}", color='red', alpha=0.7)
            axs[0].scatter(x_space, target_distribution, label="benchmark (ODE)", c='blue')
            axs[0].set_title("State distribution")
            axs[0].set_xticks(x_space)
            axs[0].set_xticklabels(labels)
            axs[0].legend()

            # Action Init
            actor_model.eval()
            with torch.no_grad():
                st_init = torch.FloatTensor(pop_init).unsqueeze(0).to(device)
                act_init = actor_model(st_init).cpu().numpy().squeeze(0)
            axs[1].bar(x_space, act_init, width=0.4, label="Action", color='#348ABD', alpha=0.7)
            axs[1].set_title("Action dist (init)")
            axs[1].set_xticks(x_space)
            axs[1].set_xticklabels(labels)
            axs[1].set_ylim([0.0, 1.0])
            axs[1].legend()

            # Action Last
            with torch.no_grad():
                st_last = torch.FloatTensor(pop_last).unsqueeze(0).to(device)
                act_last = actor_model(st_last).cpu().numpy().squeeze(0)
            axs[2].bar(x_space, act_last, width=0.4, label="Action", color='#348ABD', alpha=0.7)
            axs[2].set_title("Action dist (last)")
            axs[2].set_xticks(x_space)
            axs[2].set_xticklabels(labels)
            axs[2].set_ylim([0.0, 1.0])
            axs[2].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_eval_idist{idx}.pdf"))
            if SHOW_PLOTS:
                plt.show()
            plt.close()
            actor_model.train()

def precompute_bench_values(solver: ForwardBackwardSolver, mu_DI_vals: List[float], grid_res: int) -> List[np.ndarray]:
    """Pre-calculates Benchmark Value Function V(m) = <u(0), m> for simplex slices."""
    print(f"Pre-computing {len(mu_DI_vals)} Benchmark Simplex Slices (res={grid_res})...")
    start_t = time.time()
    x = np.linspace(0, 1.0, grid_res)
    y = np.linspace(0, 1.0, grid_res)
    X, Y = np.meshgrid(x, y)
    bench_slices = []

    for idx, di in enumerate(mu_DI_vals):
        print(f"  Computing Benchmark Slice {idx+1}/{len(mu_DI_vals)} (di={di})...")
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


def plot_value_simplex_slices(output_dir: str, prefix: str, iter_idx: int, actor: nn.Module, critic: nn.Module, device: torch.device, tables: Dict, grid_res: int = 10):
    """
    Visualizes the Value Function V(m) (Cost convention) on slices of the population simplex.
    Row 1: RL Cost (-Value = -Q(m, pi(m)))
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
    
    actor.eval()
    critic.eval()
    
    for j, di in enumerate(mu_DI_vals):
        V_rl = np.full((grid_res, grid_res), np.nan)
        V_bench = tables['bench_slices'][j]
        
        for ix in range(grid_res):
            for iy in range(grid_res):
                ds, ui = X[ix, iy], Y[ix, iy]
                if di + ds + ui <= 1.0001:
                    us = max(0, 1.0 - (di + ds + ui))
                    m = np.array([di, ds, ui, us])
                    
                    # Estimate RL Value V(m) = Q(m, actor(m))
                    # RL Cost = -V(m)
                    m_tensor = torch.FloatTensor(m).unsqueeze(0).to(device)
                    with torch.no_grad():
                        a_tensor = actor(m_tensor)
                        v_rl = critic(m_tensor, a_tensor).cpu().item()
                    V_rl[ix, iy] = -v_rl
        
        rl_costs_all.append(V_rl)
        bench_costs_all.append(V_bench)
    
    actor.train()
    critic.train()
    
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

def train_policy(env: Any,
                 actor: nn.Module, critic: nn.Module,
                 target_actor: nn.Module, target_critic: nn.Module,
                 actor_opt: optim.Optimizer, critic_opt: optim.Optimizer,
                 gamma: float, tau: float, buffer: ReplayBuffer,
                 episodes: int, plot_freq: int, print_freq: int,
                 distrib_init_sampler: Callable,
                 test_inits: List[np.ndarray], device: torch.device, output_dir: str,
                 shared_tables: Dict,
                 true_sols: Optional[List[np.ndarray]] = None,
                 window_size: int = 20):
    """
    Core training loop for the MFC reinforcement learning task.

    Args:
        env (Any): The MFC/MPC environment.
        actor (nn.Module): Actor network.
        critic (nn.Module): Critic network.
        target_actor (nn.Module): Target actor network.
        target_critic (nn.Module): Target critic network.
        actor_opt (optim.Optimizer): Optimizer for actor.
        critic_opt (optim.Optimizer): Optimizer for critic.
        gamma (float): Discount factor.
        tau (float): Soft update coefficient for target networks.
        buffer (ReplayBuffer): Transition replay buffer.
        episodes (int): Number of training episodes.
        plot_freq (int): Frequency of visualization.
        print_freq (int): Frequency of logging to console.
        distrib_init_sampler (Callable): Sampler for initial population distributions.
        test_inits (List[np.ndarray]): Fixed initial conditions for testing.
        device (torch.device): Device to use for computation.
        output_dir (str): Directory for output logs and figures.
        true_sols (Optional[List[np.ndarray]]): Benchmarks for evaluation.
        window_size (int): Window size for reward/cost moving average.
    """

    rewards = []
    eval_rewards = []
    eval_eps = []
    EVAL_EP_PER_DIST = 5 # Number of episodes to average per test distribution

    actor_lr = actor_opt.param_groups[0]['lr']
    critic_lr = critic_opt.param_groups[0]['lr']

    log_file = os.path.join(output_dir, "cybersec_training.log")
    with open(log_file, "w") as f:
        f.write("Training Log - Cyber Security\n")

    start_time = time.time()

    for ep in range(episodes + 1):
        # LR schedule
        if ep in [episodes // 3, 2 * episodes // 3]:
            actor_lr /= 2
            critic_lr /= 2
            for param_group in actor_opt.param_groups: param_group['lr'] = actor_lr
            for param_group in critic_opt.param_groups: param_group['lr'] = critic_lr
            tau /= 2

        s, _ = env.reset(distrib_init_sampler())
        traj = [s]
        tot_r = 0.0

        # Gaussian Noise decay over total_episodes
        decay_period = int(0.7 * episodes)
        start_sigma, end_sigma = 0.5, 0.05
        if ep < decay_period:
            current_sigma = start_sigma - (start_sigma - end_sigma) * (ep / decay_period)
        else:
            current_sigma = end_sigma

        step_counter = 0
        update_freq = 1

        while True:
            # Sample Gaussian noise
            noise_val = np.random.normal(0, current_sigma, size=N_STATES)
            act = get_policy_action(s, actor, noise_val, device)
            ns, r, d, _, _ = env.step(act)

            buffer.record((s, act, r, ns))
            tot_r += r
            step_counter += 1

            if buffer.counter >= buffer.batch_size and step_counter % update_freq == 0:
                bs, ba, br, bns = buffer.sample()

                # Critic update
                with torch.no_grad():
                    next_actions = target_actor(bns)
                    btq = br + gamma * target_critic(bns, next_actions)

                c_l = nn.MSELoss()(critic(bs, ba), btq)
                critic_opt.zero_grad()
                c_l.backward()
                critic_opt.step()

                # Actor update
                a_l = -critic(bs, actor(bs)).mean()
                actor_opt.zero_grad()
                a_l.backward()
                actor_opt.step()

                update_target(target_actor, actor, tau)
                update_target(target_critic, critic, tau)

            traj.append(ns)
            s = ns
            if d:
                break

        rewards.append(tot_r)

        if ep % plot_freq == 0:
            # --- Evaluation Phase ---
            all_t = [traj]
            current_eval_rs = []

            for t_init in test_inits:
                dist_rewards = []
                for _ in range(EVAL_EP_PER_DIST):
                    ts, _ = env.reset(t_init)
                    ct = [ts]
                    ep_r = 0.0
                    while True:
                        tst = torch.FloatTensor(ts).unsqueeze(0).to(device)
                        with torch.no_grad():
                            ta = [actor(tst).cpu().numpy().squeeze(0)]
                        tns, tr, td, _, _ = env.step(ta)
                        ct.append(tns)
                        ep_r += tr
                        ts = tns
                        if td: break
                    dist_rewards.append(ep_r)

                current_eval_rs.append(np.mean(dist_rewards))
                if len(all_t) <= len(test_inits): # Only store one trajectory for visualization
                    all_t.append(ct)

            mean_eval_r = np.mean(current_eval_rs)
            eval_rewards.append(mean_eval_r)
            eval_eps.append(ep)

            plot_training_progress(output_dir, "cybersec_", rewards, ep, eval_rewards, eval_eps, window_size)
            plot_evaluation(output_dir, "cybersec_", env, all_t, ep, true_sols, actor, device)
            plot_value_simplex_slices(output_dir, "cybersec_", ep, actor, critic, device, shared_tables, grid_res=10)

        if ep % print_freq == 0:
            elapsed = time.time() - start_time
            avg_r = np.mean(rewards[-window_size:]) if ep >= window_size else np.mean(rewards)
            log_str = f"Ep {ep} | Curr R: {tot_r:.4f} | Avg R ({window_size}): {avg_r:.4f} | Time: {elapsed:.2f}s"
            if len(eval_rewards) > 0 and eval_eps[-1] == ep:
                log_str += f" | Eval R: {eval_rewards[-1]:.4f}"
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")

