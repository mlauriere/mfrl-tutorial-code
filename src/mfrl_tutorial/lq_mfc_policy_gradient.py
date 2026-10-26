"""Scalar linear-quadratic MFC model and exact benchmark quantities."""

import numpy as np


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
