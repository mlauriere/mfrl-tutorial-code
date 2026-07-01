# -*- coding: utf-8 -*-
"""DDPG components for the discrete distribution-planning MFC example."""

import os
import time
from typing import Dict, List, Optional, Tuple, Any, Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

# Set seed for reproducibility
np.random.seed(1)
torch.manual_seed(1)

# Global Constants
N_STATES = 11  # Number of states
N_ACTIONS_PER_STATE = 3  # Number of actions [L, 0, R]
DISTRIBUTION_TARGET = np.asarray([0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0])

# Configuration
ACTION_UPPER_BOUND = 1.0
ACTION_LOWER_BOUND = 0.0
SHOW_PLOTS = False  # Toggle to True to display plots interactively

class DiscretePlanningEnv(gym.Env):
    """
    Custom Environment that follows gym interface for a discrete planning problem.
    """

    def __init__(self, pop_distrib: np.ndarray, T: float, dt: float, cn_sampler: Callable):
        """
        Initializes the Discrete Planning environment.

        Args:
            pop_distrib (np.ndarray): Initial population distribution over the discrete states.
            T (float): Total time horizon.
            dt (float): Time step size.
            cn_sampler (Callable): Function to sample common noise (displacement).
        """
        super(DiscretePlanningEnv, self).__init__()
        self.T = T
        self.dt = dt
        self.Nt = int(self.T / self.dt)
        self.NS = N_STATES

        self.action_space = spaces.Box(
            low=ACTION_LOWER_BOUND,
            high=ACTION_UPPER_BOUND,
            dtype=np.float64,
            shape=(self.NS, N_ACTIONS_PER_STATE)
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            dtype=np.float64,
            shape=(self.NS,)
        )

        self.pop_distrib_discrete_init = pop_distrib.copy()
        self.cn_traj = []
        self.cn_sampler = cn_sampler
        self.reset(self.pop_distrib_discrete_init)

    def running_cost_t_pop(self, mu: np.ndarray, action: np.ndarray) -> float:
        """
        Calculates the immediate social cost for the entire population.

        Args:
            mu (np.ndarray): Current population distribution.
            action (np.ndarray): Actions (matrix of move probabilities for each state).

        Returns:
            float: The calculated social cost.
        """
        distrib_cost = np.sqrt(np.sum((mu - DISTRIBUTION_TARGET)**2))
        action_cost = np.dot(mu, (action[:, 0] + action[:, 2]))
        coef_distrib_cost = 2.0
        return coef_distrib_cost * distrib_cost + action_cost

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Transitions the population and calculates the reward.

        Args:
            action (np.ndarray): Actions selected by the policy.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: (next_state, reward, done, truncated, info).
        """
        # Handle formatting differences
        if isinstance(action, (list, tuple)):
            action = action[0]

        mu = self.state
        social_reward = -self.running_cost_t_pop(mu, action)

        new_mu = np.zeros(self.NS)
        for i_S in range(1, self.NS - 1):
            new_mu[i_S - 1] += mu[i_S] * action[i_S][0]
            new_mu[i_S] += mu[i_S] * action[i_S][1]
            new_mu[i_S + 1] += mu[i_S] * action[i_S][2]

        i_S_fixed = self.NS - 2
        new_mu[0] += mu[0] * action[i_S_fixed][0]
        new_mu[0] += mu[0] * action[i_S_fixed][1]
        new_mu[1] += mu[0] * action[i_S_fixed][2]

        new_mu[self.NS - 2] += mu[self.NS - 1] * action[i_S_fixed][0]
        new_mu[self.NS - 1] += mu[self.NS - 1] * action[i_S_fixed][1]
        new_mu[self.NS - 1] += mu[self.NS - 1] * action[i_S_fixed][2]

        cn_val = self.cn_sampler()
        self.cn_traj = np.append(self.cn_traj, cn_val)

        new_mu = np.roll(new_mu, int(cn_val))
        if cn_val == 1:
            new_mu[N_STATES - 1] += new_mu[0]
            new_mu[0] = 0
        elif cn_val == -1:
            new_mu[0] += new_mu[N_STATES - 1]
            new_mu[N_STATES - 1] = 0

        self.state = new_mu
        self.t += self.dt
        done = (self.t > self.T - 0.5 * self.dt)
        return self.state, social_reward, done, False, {}

    def reset(self, pop_distrib: Optional[np.ndarray] = None, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.t = 0.0
        if pop_distrib is not None:
            self.state = pop_distrib
        else:
            self.state = self.pop_distrib_discrete_init
        self.cn_traj = []
        return self.state, {}


class ReplayBuffer:
    """Experience replay buffer for DDPG, storing PyTorch tensors."""
    def __init__(self, num_states: int, action_shape: Tuple, device: torch.device, capacity: int = 100000, batch_size: int = 64):
        self.capacity, self.batch_size, self.counter = capacity, batch_size, 0
        self.device = device
        self.s_buf = torch.zeros((capacity, num_states), dtype=torch.float32, device=device)
        self.a_buf = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=device)
        self.r_buf = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.ns_buf = torch.zeros((capacity, num_states), dtype=torch.float32, device=device)

    def record(self, obs: Tuple):
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
    Actor network for Discrete Planning MFC Policy.
    """
    def __init__(self, ns: int, n_states: int, na_per_state: int, hidden_dims: List[int] = [64, 64], activation: Any = nn.ReLU):
        """
        Initializes the Actor network with configurable layers.

        Args:
            ns (int): Number of states in the mean field.
            n_states (int): Number of physical states.
            na_per_state (int): Number of actions per state.
            hidden_dims (List[int]): List of hidden layer dimensions.
            activation (Any): Activation function class to use between layers.
        """
        super(Actor, self).__init__()
        self.n_states = n_states
        self.na_per_state = na_per_state

        layers = []
        last_dim = ns
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(activation())
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, n_states * na_per_state))
        self.net = nn.Sequential(*layers)

        # Initialize final layer weights to be small
        nn.init.uniform_(self.net[-1].weight, -0.003, 0.003)
        nn.init.uniform_(self.net[-1].bias, -0.003, 0.003)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Actor. Outputs action probabilities per state.
        """
        out = self.net(x)
        out = 10.0 * torch.tanh(out)
        out = out.view(-1, self.n_states, self.na_per_state)
        return torch.softmax(out, dim=2)


class Critic(nn.Module):
    """
    Critic network for Discrete Planning MFC Value estimation.
    """
    def __init__(self, ns: int, na_total: int,
                 state_hidden_dims: List[int] = [64],
                 action_hidden_dims: List[int] = [64],
                 combined_hidden_dims: List[int] = [64],
                 activation: Any = nn.ReLU):
        """
        Initializes the Critic network with configurable branches.

        Args:
            ns (int): Number of states in the mean field.
            na_total (int): Total action dimension.
            state_hidden_dims (List[int]): Hidden layers for state branch.
            action_hidden_dims (List[int]): Hidden layers for action branch.
            combined_hidden_dims (List[int]): Hidden layers for combined branch.
            activation (Any): Activation function class.
        """
        super(Critic, self).__init__()

        # State branch
        s_layers = []
        last_s = ns
        for h in state_hidden_dims:
            s_layers.append(nn.Linear(last_s, h))
            s_layers.append(activation())
            last_s = h
        self.state_branch = nn.Sequential(*s_layers)

        # Action branch
        a_layers = []
        last_a = na_total
        for h in action_hidden_dims:
            a_layers.append(nn.Linear(last_a, h))
            a_layers.append(activation())
            last_a = h
        self.action_branch = nn.Sequential(*a_layers)

        # Combined branch
        c_layers = []
        last_c = last_s + last_a
        for h in combined_hidden_dims:
            c_layers.append(nn.Linear(last_c, h))
            c_layers.append(activation())
            last_c = h
        c_layers.append(nn.Linear(last_c, 1))
        self.combined = nn.Sequential(*c_layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Critic with dual branching.
        """
        s_out = self.state_branch(s)
        a_flat = a.view(a.size(0), -1)
        a_out = self.action_branch(a_flat)
        combined = torch.cat([s_out, a_out], dim=1)
        return self.combined(combined)


def update_target(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(sp.data * tau + tp.data * (1 - tau))


def get_policy_action(state: np.ndarray, actor: nn.Module, noise: np.ndarray, device: torch.device) -> List[np.ndarray]:
    """
    Selects an action from the actor policy, adds noise, and ensures it's a valid probability distribution.

    Args:
        state (np.ndarray): Current population distribution.
        actor (nn.Module): Policy network.
        noise (np.ndarray): Gaussian noise matrix.
        device (torch.device): Computation device.
        List[np.ndarray]: List containing the normalized action matrix.
    """
    st = torch.FloatTensor(state).unsqueeze(0).to(device)
    actor.eval()
    with torch.no_grad():
        a = actor(st).cpu().numpy().squeeze(0)
    actor.train()

    noisy_a = np.clip(a + noise, 0, 1)
    norm = np.sum(noisy_a, axis=1, keepdims=True)
    legal_a = noisy_a / (norm + 1e-8)
    return [legal_a]


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

    ax.set_xlabel("Episodes", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, ls='--', alpha=0.6)
    ax.legend(loc='best', fontsize='large')
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

    ax.set_xlabel("Episodes", fontsize=12)
    ax.set_ylabel("Total Cost", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, ls='--', alpha=0.6)
    ax.legend(loc='best', fontsize='large')
    plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_training_cost.pdf"), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_evaluation(output_dir: str, prefix: str, env: Any, trajs: List[List[np.ndarray]], ep: int, actor_model: nn.Module, device: torch.device, window_size: int = 20):
    """
    Generates evaluation plots including state distributions, actions, and evolution over time.

    Args:
        output_dir (str): Directory to save the plots.
        prefix (str): Filename prefix.
        env (Any): Simulation environment.
        trajs (List[List[np.ndarray]]): List of trajectories (distributions over time).
        ep (int): Current episode.
        actor_model (nn.Module): Policy model to visualize action distributions.
        device (torch.device): Device for actor inference.
        window_size (int): Window size for averaging state distributions.
    """
    n_tests = len(trajs) - 1
    if n_tests <= 0: return

    t_space = np.linspace(0, env.T, num=env.Nt + 1)

    fig_test, axes = plt.subplots(1, n_tests, figsize=(4 * n_tests, 3.5), squeeze=False)
    for idx in range(n_tests):
        ax_t = axes[0, idx]
        data = np.asarray(trajs[idx + 1])
        colors = cm.viridis(np.linspace(0, 1, N_STATES))

        # Plot target (dashed) first with larger width
        for i in range(N_STATES):
            ax_t.axhline(y=DISTRIBUTION_TARGET[i], ls='--', lw=3.0, color=colors[i], alpha=0.4)

        # Plot RL trajectories (full) on top
        for i in range(N_STATES):
            ax_t.plot(t_space, data[:, i], lw=2.0, color=colors[i], alpha=0.9)

        ax_t.set_xlabel("Time (t)", fontsize=12)
        if idx == 0:
            ax_t.set_ylabel("Proportion", fontsize=12)
        ax_t.set_ylim(0, 1.05)
        ax_t.tick_params(axis='both', which='major', labelsize=10)
        ax_t.set_title(f"Test Init {idx}", fontsize=14)
        ax_t.grid(True, ls='--', alpha=0.5)
        # Add legend if it's the first plot or multiple states need labels
        labels = [f"S{i}" for i in range(N_STATES)]
        ax_t.legend(labels, loc='best', fontsize='medium')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_eval.pdf"), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # --- Part 2: Detailed Per-Distribution View (Enhanced v4 Style) ---
    for idx in range(n_tests):
        plt.close('all')
        x_space = np.linspace(1, N_STATES, num=N_STATES, endpoint=True)
        pop_list = trajs[idx + 1]

        pop_init = np.asarray(pop_list[0])
        pop_last = np.asarray(pop_list[-1])
        pop_avg = np.mean(pop_list[-window_size:], axis=0) if len(pop_list) >= window_size else pop_last

        # Now 5 subplots: Init vs Target, Final vs Target, Action Init, Action Last, Common Noise
        fig, axs = plt.subplots(1, 5, figsize=(25, 4))

        # 1. State Distribution: Init vs Target
        # Target as background
        axs[0].fill_between(x_space, 0, DISTRIBUTION_TARGET, facecolor='blue', alpha=0.1, label="Target Area")
        axs[0].plot(x_space, DISTRIBUTION_TARGET, color='blue', ls='--', alpha=0.3, label="Target")
        # Init as line
        axs[0].plot(x_space, pop_init, marker='o', color='green', label="Initial", lw=2)
        axs[0].set_title("State: Init vs Target")
        axs[0].set_xlabel("State Index")
        axs[0].set_ylabel("Proportion")
        axs[0].set_ylim(0, 0.45)
        axs[0].legend()

        # 2. State Distribution: Final / Average vs Target
        # Target as background
        axs[1].fill_between(x_space, 0, DISTRIBUTION_TARGET, facecolor='blue', alpha=0.1)
        axs[1].plot(x_space, DISTRIBUTION_TARGET, color='blue', ls='--', alpha=0.3)
        # Average with shading
        axs[1].fill_between(x_space, 0, pop_avg, facecolor='red', alpha=0.1)
        axs[1].plot(x_space, pop_avg, marker='s', color='#E24A33', label=f"Avg (last {window_size})", lw=1.5, alpha=0.6)
        # Last as solid line
        axs[1].plot(x_space, pop_last, marker='*', color='purple', label="Last Episode", lw=2)
        axs[1].set_title("State: Final/Avg vs Target")
        axs[1].set_xlabel("State Index")
        axs[1].legend()

        # 3. Action Init (Stacked)
        actor_model.eval()
        with torch.no_grad():
            st_init = torch.FloatTensor(pop_init).unsqueeze(0).to(device)
            act_init = actor_model(st_init).cpu().numpy().squeeze(0)

        axs[2].bar(x_space, act_init[:, 0], label="Left", alpha=0.7, color='#348ABD')
        axs[2].bar(x_space, act_init[:, 1], bottom=act_init[:, 0], label="Stay", alpha=0.7, color='#7A68A6')
        axs[2].bar(x_space, act_init[:, 2], bottom=act_init[:, 0] + act_init[:, 1], label="Right", alpha=0.7, color='#A60628')
        axs[2].set_title("Action dist (init)")
        axs[2].set_xlabel("State Index")
        axs[2].set_ylim(0, 1.1)
        axs[2].legend(loc='upper right', fontsize='small')

        # 4. Action Last (Stacked)
        with torch.no_grad():
            st_last = torch.FloatTensor(pop_last).unsqueeze(0).to(device)
            act_last = actor_model(st_last).cpu().numpy().squeeze(0)

        axs[3].bar(x_space, act_last[:, 0], label="Left", alpha=0.7, color='#348ABD')
        axs[3].bar(x_space, act_last[:, 1], bottom=act_last[:, 0], label="Stay", alpha=0.7, color='#7A68A6')
        axs[3].bar(x_space, act_last[:, 2], bottom=act_last[:, 0] + act_last[:, 1], label="Right", alpha=0.7, color='#A60628')
        axs[3].set_title("Action dist (last)")
        axs[3].set_xlabel("State Index")
        axs[3].set_ylim(0, 1.1)
        axs[3].legend(loc='upper right', fontsize='small')

        # 5. Common Noise
        t_noise = np.arange(len(env.cn_traj)) * env.dt
        axs[4].plot(t_noise, np.cumsum(env.cn_traj), label="Cumul CN", color='black')
        axs[4].set_title("Common noise")
        axs[4].set_xlabel("Time")
        axs[4].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_eval_idist{idx}.pdf"))
        if SHOW_PLOTS:
            plt.show()
        plt.close()
        actor_model.train()

    # --- Part 3: Evolution Plot (Gradient View) ---
    fig_evol, axes_evol = plt.subplots(1, n_tests, figsize=(5 * n_tests, 5), squeeze=False)
    colors = cm.viridis(np.linspace(0, 1, N_STATES))
    for idx in range(n_tests):
        ax_e = axes_evol[0, idx]
        data = np.asarray(trajs[idx + 1])
        for i in range(N_STATES):
            ax_e.plot(t_space, data[:, i], lw=2, color=colors[i], label=f"State {i}")
            ax_e.axhline(y=DISTRIBUTION_TARGET[i], ls='--', color=colors[i], alpha=0.5)
        ax_e.set_title(f"Evolution - Init {idx}")
        ax_e.set_xlabel("Time")
        ax_e.set_ylabel("Distribution")
        ax_e.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_eval_evolution.pdf"), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # --- Part 4: Snapshot Evolution View (Geometric/Log Spacing) ---
    snapshot_indices = [0, 2, 5, 10, 25, env.Nt] 
    actor_model.eval()
    for idx in range(n_tests):
        pop_list = trajs[idx + 1]
        x_space = np.linspace(1, N_STATES, num=N_STATES, endpoint=True)
        
        # Use 2 rows: Top for population, Bottom for actions
        fig, axs = plt.subplots(2, len(snapshot_indices), figsize=(4 * len(snapshot_indices), 7.0), sharex='col')
        for i, t_idx in enumerate(snapshot_indices):
            # Row 0: Population Distribution
            ax_p = axs[0, i]
            actual_idx = min(t_idx, len(pop_list) - 1)
            pop_t = np.asarray(pop_list[actual_idx])
            t_val = actual_idx * env.dt
            
            # Target Background
            ax_p.fill_between(x_space, 0, DISTRIBUTION_TARGET, facecolor='blue', alpha=0.1)
            ax_p.plot(x_space, DISTRIBUTION_TARGET, color='blue', ls='--', alpha=0.2, label="Target" if i==0 else None)
            
            # Population at time t
            ax_p.plot(x_space, pop_t, marker='o', color='#E24A33', lw=2, markersize=4, label="Current" if i==0 else None)
            
            ax_p.set_title(f"Time t={t_val:.1f}", fontsize=15, fontweight='bold')
            ax_p.tick_params(axis='both', which='major', labelsize=12)
            if i == 0:
                ax_p.set_ylabel("Proportion", fontsize=14)
                ax_p.legend(loc='upper center', ncol=2, fontsize=11, frameon=False)
            ax_p.set_ylim(0, 0.5)
            ax_p.grid(True, alpha=0.2)

            # Row 1: Action Distribution (Stacked Bar)
            ax_a = axs[1, i]
            with torch.no_grad():
                st_t = torch.FloatTensor(pop_t).unsqueeze(0).to(device)
                act_t = actor_model(st_t).cpu().numpy().squeeze(0)
            
            ax_a.bar(x_space, act_t[:, 0], label="Left", alpha=0.7, color='#348ABD')
            ax_a.bar(x_space, act_t[:, 1], bottom=act_t[:, 0], label="Stay", alpha=0.7, color='#7A68A6')
            ax_a.bar(x_space, act_t[:, 2], bottom=act_t[:, 0] + act_t[:, 1], label="Right", alpha=0.7, color='#A60628')
            
            ax_a.set_xlabel("State Index", fontsize=14)
            ax_a.tick_params(axis='both', which='major', labelsize=12)
            if i == 0:
                ax_a.set_ylabel("Action Prob", fontsize=14)
                # Legend as a row inside the plot
                ax_a.legend(loc='upper center', ncol=3, fontsize=11, frameon=False)
            ax_a.set_ylim(0, 1.2)
            ax_a.grid(True, alpha=0.2)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}ep{ep}_eval_snapshots_idist{idx}.pdf"), bbox_inches='tight')
        plt.close(fig)
    actor_model.train()


def train_policy(env: Any,
                 actor: nn.Module, critic: nn.Module,
                 target_actor: nn.Module, target_critic: nn.Module,
                 actor_opt: optim.Optimizer, critic_opt: optim.Optimizer,
                 gamma: float, tau: float, buffer: ReplayBuffer,
                 episodes: int, plot_freq: int, print_freq: int,
                 distrib_init_sampler: Callable,
                 test_inits: List[np.ndarray], device: torch.device, output_dir: str,
                 true_sols: Optional[List[np.ndarray]] = None,
                 save_freq: int = 500,
                 window_size: int = 20):
    """
    Core training loop for the Discrete Planning MFC reinforcement learning task.

    Args:
        env (Any): The Discrete Planning MFC environment.
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
        true_sols (Optional[List[np.ndarray]]): Optional benchmarks (not typically used in this script).
        window_size (int): Window size for reward/cost moving average.
    """

    rewards = []
    eval_rewards = []
    eval_eps = []
    EVAL_EP_PER_DIST = 5 # Number of episodes to average per test distribution

    actor_lr = actor_opt.param_groups[0]['lr']
    critic_lr = critic_opt.param_groups[0]['lr']

    log_file = os.path.join(output_dir, "distribplanning_training.log")
    with open(log_file, "w") as f:
        f.write("Training Log - Discrete Planning\n")

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
        decay_period = int(0.7 * episodes)  # Decay over 70% of total episodes
        start_sigma, end_sigma = 0.2, 0.01
        if ep < decay_period:
            current_sigma = start_sigma - (start_sigma - end_sigma) * (ep / decay_period)
        else:
            current_sigma = end_sigma

        step_counter = 0
        update_freq = 1
        num_updates_per_step = 5

        while True:
            # Sample Gaussian noise for current policy action
            noise_val = np.random.normal(0, current_sigma, size=(N_STATES, N_ACTIONS_PER_STATE))
            act = get_policy_action(s, actor, noise_val, device)
            ns, r, done, _, _ = env.step(act)

            # Reward scaling for stronger gradient signal
            r_scaled = r * 10.0
            buffer.record((s, act, r_scaled, ns))
            tot_r += r # Log the original unscaled reward
            step_counter += 1

            if buffer.counter >= buffer.batch_size and step_counter % update_freq == 0:
                for _ in range(num_updates_per_step):
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
            if done:
                break

        rewards.append(tot_r)

        if ep % plot_freq == 0 or ep % save_freq == 0:
            # --- Evaluation Phase ---
            all_t = [traj]
            current_eval_rs = []
            test_data_for_saving = []

            for t_init in test_inits:
                dist_rewards = []
                for j in range(EVAL_EP_PER_DIST):
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
                        if td:
                            break
                    dist_rewards.append(ep_r)
                    if j == 0:
                        # Capture data for visualization and saving
                        test_run_info = {
                            'traj': np.array(ct),
                            'cn_traj': np.array(env.cn_traj)
                        }
                        test_data_for_saving.append(test_run_info)
                        all_t.append(ct)

                current_eval_rs.append(np.mean(dist_rewards))

            mean_eval_r = np.mean(current_eval_rs)
            eval_rewards.append(mean_eval_r)
            eval_eps.append(ep)

            if ep % plot_freq == 0:
                plot_training_progress(output_dir, "distribplanning_", rewards, ep, eval_rewards, eval_eps, window_size)
                plot_evaluation(output_dir, "distribplanning_", env, all_t, ep, target_actor, device, window_size)

            if ep % save_freq == 0:
                data_dir = os.path.join(output_dir, "data")
                os.makedirs(data_dir, exist_ok=True)
                
                # Add action data to test_data_for_saving
                for i in range(len(test_inits)):
                    pop_list = test_data_for_saving[i]['traj']
                    with torch.no_grad():
                        st_init = torch.FloatTensor(pop_list[0]).unsqueeze(0).to(device)
                        act_init = actor(st_init).cpu().numpy().squeeze(0)
                        st_last = torch.FloatTensor(pop_list[-1]).unsqueeze(0).to(device)
                        act_last = actor(st_last).cpu().numpy().squeeze(0)
                    test_data_for_saving[i]['act_init'] = act_init
                    test_data_for_saving[i]['act_last'] = act_last

                save_dict = {
                    'ep': ep,
                    'rewards': np.array(rewards),
                    'eval_rewards': np.array(eval_rewards),
                    'eval_eps': np.array(eval_eps),
                    'test_data': test_data_for_saving,
                    'target_dist': DISTRIBUTION_TARGET
                }
                np.save(os.path.join(data_dir, f"data_ep{ep}.npy"), save_dict)
                print(f"Data saved at episode {ep}")

        if ep % print_freq == 0:
            elapsed = time.time() - start_time
            avg_r = np.mean(rewards[-window_size:]) if ep >= window_size else np.mean(rewards)
            log_str = f"Ep {ep} | Curr R: {tot_r:.4f} | Avg R ({window_size}): {avg_r:.4f} | Time: {elapsed:.2f}s"
            if len(eval_rewards) > 0 and eval_eps[-1] == ep:
                log_str += f" | Eval R: {eval_rewards[-1]:.4f}"
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")

