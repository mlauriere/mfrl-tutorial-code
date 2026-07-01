# -*- coding: utf-8 -*-
"""Policy-gradient solvers for the scalar linear-quadratic MFC example."""

import os

import numpy as np
import matplotlib.pyplot as plt

# --- 1. MODEL & OPTIMIZERS ---

class LQMFCModel:
    def __init__(self, params):
        self.a, self.a_bar = params['a'], params['a_bar']
        self.b, self.b_bar = params['b'], params['b_bar']
        self.q, self.q_bar = params['q'], params['q_bar']
        self.r, self.r_bar = params['r'], params['r_bar']
        self.gamma = params['gamma']
        self.A, self.B, self.Q, self.R = self.a, self.b, self.q, self.r
        self.A_tilde, self.B_tilde = self.a + self.a_bar, self.b + self.b_bar
        self.Q_tilde, self.R_tilde = self.q + self.q_bar, self.r + self.r_bar
        self.var_y0 = self.var_z0 = 1/3
        self.var_eps = self.var_eps0 = 0.01

    def solve_riccati(self, tol=1e-12, max_iter=1000):
        P = self.Q
        for _ in range(max_iter):
            P_next = self.Q + self.gamma * self.A**2 * P - \
                     (self.gamma**2 * self.A**2 * self.B**2 * P**2) / (self.R + self.gamma * self.B**2 * P)
            if abs(P_next - P) < tol: break
            P = P_next
        K_star = (self.gamma * self.A * self.B * P) / (self.R + self.gamma * self.B**2 * P)

        P_t = self.Q_tilde
        for _ in range(max_iter):
            P_t_next = self.Q_tilde + self.gamma * self.A_tilde**2 * P_t - \
                       (self.gamma**2 * self.A_tilde**2 * self.B_tilde**2 * P_t**2) / (self.R_tilde + self.gamma * self.B_tilde**2 * P_t)
            if abs(P_t_next - P_t) < tol: break
            P_t = P_t_next
        L_star = (self.gamma * self.A_tilde * self.B_tilde * P_t) / (self.R_tilde + self.gamma * self.B_tilde**2 * P_t)
        return K_star, L_star

    def compute_cost(self, K, L):
        rho_y = self.gamma * (self.A - self.B * K)**2
        rho_z = self.gamma * (self.A_tilde - self.B_tilde * L)**2
        if rho_y >= 1.0 or rho_z >= 1.0: return 1e6
        sigma_k = (self.var_y0 + (self.gamma * self.var_eps) / (1 - self.gamma)) / (1 - rho_y)
        sigma_l = (self.var_z0 + (self.gamma * self.var_eps0) / (1 - self.gamma)) / (1 - rho_z)
        return (self.Q + K**2 * self.R) * sigma_k + (self.Q_tilde + L**2 * self.R_tilde) * sigma_l

    def compute_exact_gradients(self, K, L):
        rho_y = self.gamma * (self.A - self.B * K)**2
        rho_z = self.gamma * (self.A_tilde - self.B_tilde * L)**2
        sigma_k = (self.var_y0 + (self.gamma * self.var_eps) / (1 - self.gamma)) / (1 - rho_y)
        sigma_l = (self.var_z0 + (self.gamma * self.var_eps0) / (1 - self.gamma)) / (1 - rho_z)
        pk = (self.Q + K**2 * self.R) / (1 - rho_y)
        pl = (self.Q_tilde + L**2 * self.R_tilde) / (1 - rho_z)
        gk = 2 * ((self.R + self.gamma * self.B**2 * pk) * K - self.gamma * self.A * self.B * pk) * sigma_k
        gl = 2 * ((self.R_tilde + self.gamma * self.B_tilde**2 * pl) * L - self.gamma * self.A_tilde * self.B_tilde * pl) * sigma_l
        return gk, gl

    def simulator_mkv(self, K, L, T):
        y, z = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        total_cost = 0.0
        for t in range(T):
            total_cost += (self.gamma**t) * (self.q * (y**2) + self.r * ((-K * y)**2) + (self.q + self.q_bar) * (z**2) + (self.r + self.r_bar) * ((-L * z)**2))
            y = (self.A - self.B * K) * y + np.random.normal(0, np.sqrt(self.var_eps))
            z = (self.A_tilde - self.B_tilde * L) * z + np.random.normal(0, np.sqrt(self.var_eps0))
        return total_cost

    def simulator_pop(self, K, L, T, N):
        x = np.random.uniform(-1, 1, N) + np.random.uniform(-1, 1)
        total_social_cost = 0.0
        for t in range(T):
            x_bar = np.mean(x)
            u = -K * (x - x_bar) - L * x_bar
            u_bar = np.mean(u)
            costs = self.q * (x**2) + self.q_bar * (x_bar**2) + self.r * (u**2) + self.r_bar * (u_bar**2)
            total_social_cost += (self.gamma**t) * np.mean(costs)
            x = self.a * x + self.a_bar * x_bar + self.b * u + self.b_bar * u_bar + \
                np.random.normal(0, np.sqrt(self.var_eps), N) + np.random.normal(0, np.sqrt(self.var_eps0))
        return total_social_cost

class AdamOptimizer:
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.eta, self.beta1, self.beta2, self.epsilon = eta, beta1, beta2, epsilon
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        if self.m is None:
            self.m, self.v = np.zeros_like(params), np.zeros_like(params)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params - self.eta * m_hat / (np.sqrt(v_hat) + self.epsilon)

# --- 2. SOLVERS ---

class BaseSolver:
    def __init__(self, optimizer): self.opt = optimizer
    def update(self, K, L): raise NotImplementedError()

class ExactPGSolver(BaseSolver):
    def __init__(self, model, optimizer):
        super().__init__(optimizer)
        self.model = model
    def update(self, K, L):
        gk, gl = self.model.compute_exact_gradients(K, L)
        new = self.opt.step(np.array([K, L]), np.array([gk, gl]))
        return new[0], new[1]

class ModelFreeMKVSolver(BaseSolver):
    def __init__(self, model, optimizer, T, M, tau):
        super().__init__(optimizer)
        self.model, self.T, self.M, self.tau = model, T, M, tau
    def update(self, K, L):
        sk, sl = 0.0, 0.0
        for _ in range(self.M):
            vk, vl = self.tau * np.random.choice([-1, 1]), self.tau * np.random.choice([-1, 1])
            Ci = self.model.simulator_mkv(K + vk, L + vl, self.T)
            sk += Ci * vk; sl += Ci * vl
        gk, gl = sk / (self.tau**2 * self.M), sl / (self.tau**2 * self.M)
        new = self.opt.step(np.array([K, L]), np.array([gk, gl]))
        return new[0], new[1]

class ModelFreePopSolver(BaseSolver):
    def __init__(self, model, optimizer, T, N, M, tau):
        super().__init__(optimizer)
        self.model, self.T, self.N, self.M, self.tau = model, T, N, M, tau
    def update(self, K, L):
        sk, sl = 0.0, 0.0
        for _ in range(self.M):
            vk, vl = self.tau * np.random.choice([-1, 1]), self.tau * np.random.choice([-1, 1])
            Ci = self.model.simulator_pop(K + vk, L + vl, self.T, self.N)
            sk += Ci * vk; sl += Ci * vl
        gk, gl = sk / (self.tau**2 * self.M), sl / (self.tau**2 * self.M)
        new = self.opt.step(np.array([K, L]), np.array([gk, gl]))
        return new[0], new[1]

# --- 3. PLOTTING UTILS ---

def apply_style(ax, title, ylabel=None, is_log=False):
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iterations', fontsize=16)
    if ylabel: ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if is_log: ax.set_yscale('log')

def _plot_data_on_axes(ax_dist, ax_K, ax_L, ax_cost, ax_cost_gap, h, s, label, plot_legend_label, opt_cost=None):
    legend_label = label if plot_legend_label else "_nolegend_"
    if 'K_mean' in h:
        if ax_dist:
            dist_mean, dist_std = np.array(h['dist_mean']), np.array(h['dist_std'])
            ax_dist.plot(dist_mean, **s, label=legend_label)
            ax_dist.fill_between(range(len(dist_mean)), np.maximum(dist_mean - dist_std, 1e-15), dist_mean + dist_std, color=s['color'], alpha=0.2)
        if ax_K:
            K_mean, K_std = np.array(h['K_mean']), np.array(h['K_std'])
            ax_K.plot(K_mean, **s, label=legend_label)
            ax_K.fill_between(range(len(K_mean)), K_mean - K_std, K_mean + K_std, color=s['color'], alpha=0.2)
        if ax_L:
            L_mean, L_std = np.array(h['L_mean']), np.array(h['L_std'])
            ax_L.plot(L_mean, **s, label=legend_label)
            ax_L.fill_between(range(len(L_mean)), L_mean - L_std, L_mean + L_std, color=s['color'], alpha=0.2)
        if ax_cost:
            cost_mean, cost_std = np.array(h['cost_mean']), np.array(h['cost_std'])
            ax_cost.plot(cost_mean, **s, label=legend_label)
            ax_cost.fill_between(range(len(cost_mean)), cost_mean - cost_std, cost_mean + cost_std, color=s['color'], alpha=0.2)
        if ax_cost_gap and opt_cost is not None:
            gap_mean, gap_std = np.array(h['cost_mean']) - opt_cost, np.array(h['cost_std'])
            ax_cost_gap.plot(np.maximum(gap_mean, 1e-15), **s, label=legend_label)
            ax_cost_gap.fill_between(range(len(gap_mean)), np.maximum(gap_mean - gap_std, 1e-15), np.maximum(gap_mean + gap_std, 1e-15), color=s['color'], alpha=0.2)
    else:
        if ax_dist: ax_dist.plot(h['dist'], **s, label=legend_label)
        if ax_K: ax_K.plot(h['K'], **s, label=legend_label)
        if ax_L: ax_L.plot(h['L'], **s, label=legend_label)
        if ax_cost: ax_cost.plot(h['cost'], **s, label=legend_label)
        if ax_cost_gap and opt_cost is not None: ax_cost_gap.plot(np.maximum(np.array(h['cost']) - opt_cost, 1e-15), **s, label=legend_label)

def plot_layout(h_list, labels, styles, K_star, L_star, opt_cost, main_title, output_dir=None, filename_prefix="", show_plots=False):
    is_comparison = "Comparison" in main_title
    plot_legend_label = True

    fig_params, axs_params = plt.subplots(1, 3, figsize=(18, 6))
    for h, label in zip(h_list, labels):
        # Improved style lookup: try full label first, then base name
        s = styles.get(label, styles.get(label.split(' (')[0], {'color': None}))
        _plot_data_on_axes(axs_params[0], axs_params[1], axs_params[2], None, None, h, s, label, plot_legend_label, opt_cost)
    apply_style(axs_params[0], 'Param Dist (Log)', ylabel='Distance to Optimal', is_log=True)
    apply_style(axs_params[1], 'K', ylabel='K Value'); axs_params[1].axhline(y=K_star, color='red', ls='--', alpha=0.6, label='Optimal K')
    apply_style(axs_params[2], 'L', ylabel='L Value'); axs_params[2].axhline(y=L_star, color='red', ls='--', alpha=0.6, label='Optimal L')
    for ax in axs_params: ax.legend(fontsize=14)
    plt.tight_layout()
    if output_dir: fig_params.savefig(os.path.join(output_dir, f"{filename_prefix}_params.pdf"), bbox_inches='tight')
    if show_plots: plt.show()
    plt.close(fig_params)

    fig_costs, axs_costs = plt.subplots(1, 2, figsize=(12, 6))
    for h, label in zip(h_list, labels):
        s = styles.get(label, styles.get(label.split(' (')[0], {'color': None}))
        _plot_data_on_axes(None, None, None, axs_costs[0], axs_costs[1], h, s, label, plot_legend_label, opt_cost)
    apply_style(axs_costs[0], 'Social Cost', ylabel='Cost Value'); axs_costs[0].axhline(y=opt_cost, color='red', ls='--', alpha=0.6, label='Optimal Cost')
    apply_style(axs_costs[1], 'Cost Gap (Log)', ylabel='Cost Difference', is_log=True); axs_costs[1].set_ylim(bottom=1e-8)
    for ax in axs_costs: ax.legend(fontsize=14)
    plt.tight_layout()
    if output_dir: fig_costs.savefig(os.path.join(output_dir, f"{filename_prefix}_costs.pdf"), bbox_inches='tight')
    if show_plots: plt.show()
    plt.close(fig_costs)

# --- 4. EXECUTION ENGINE ---

def run_single_solver(model, solver_class, solver_args, n_iter, seed, output_dir, label, styles, Ks, Ls, oc, show_plots, plot_every):
    if seed is not None: np.random.seed(seed)
    
    # Check if data already exists to support "resume" behavior
    data_path = os.path.join(output_dir, "data", f"{label.replace(' ', '_').lower()}.npy")
    if os.path.exists(data_path):
        print(f"  [Skipping] Data for {label} already exists.")
        return np.load(data_path, allow_pickle=True).item()

    optimizer = AdamOptimizer(**solver_args['opt_kwargs'])
    solver = solver_class(model, optimizer, **solver_args['solver_kwargs'])
    
    K, L = 0.0, 0.0
    h = {'K': [], 'L': [], 'cost': [], 'dist': []}
    
    for i in range(1, n_iter + 1):
        c = model.compute_cost(K, L)
        h['K'].append(K); h['L'].append(L); h['cost'].append(c)
        h['dist'].append(np.sqrt((K-Ks)**2 + (L-Ls)**2))
        K, L = solver.update(K, L)
        
        if i % plot_every == 0:
            plot_layout([h], [label], styles, Ks, Ls, oc, f"Progress: {label}", output_dir=output_dir, filename_prefix=f"progress_{label.replace(' ', '_').lower()}", show_plots=show_plots)

    # Final individual plot for this run
    plot_layout([h], [label], styles, Ks, Ls, oc, f"Final Result: {label}", output_dir=output_dir, filename_prefix=f"final_{label.replace(' ', '_').lower()}", show_plots=show_plots)

    # Save immediately after finishing
    np.save(data_path, h)
    return h
