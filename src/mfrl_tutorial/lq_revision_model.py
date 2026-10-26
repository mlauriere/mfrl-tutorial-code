"""Model adapters and exact saved-policy evaluation; no training is launched."""
import numpy as np


def make_model(config):
    if config['dimension'] == 1:
        from .lq_mfc_policy_gradient import LQMFCModel
        model = LQMFCModel(config['model'])
        model.var_eps = model.var_eps0 = config['model']['innovation_covariance_per_component']
        return model
    from .lq_revision_matrix_reference import Model
    return Model(config['model'])


def optimum(model, dimension):
    return np.asarray(model.solve_riccati() if dimension == 1 else model.optimum)


def exact_gradient(model, theta, dimension):
    return np.asarray(model.compute_exact_gradients(*theta) if dimension == 1 else model.cost(theta, gradient=True)[1])


def cost_parts(model, theta, dimension):
    """Infinite-horizon centered/mean costs; accepts leading batch dimensions."""
    theta = np.asarray(theta)
    if dimension == 1:
        K, L = theta[..., 0], theta[..., 1]
        ry, rz = model.gamma * (model.A-model.B*K)**2, model.gamma * (model.A_tilde-model.B_tilde*L)**2
        if max(float(np.max(ry)), float(np.max(rz))) >= 1:
            raise ValueError('Discounted unstable scalar gain')
        cy = (model.Q+model.R*K*K)*(model.var_y0+model.gamma*model.var_eps/(1-model.gamma))/(1-ry)
        cz = (model.Q_tilde+model.R_tilde*L*L)*(model.var_z0+model.gamma*model.var_eps0/(1-model.gamma))/(1-rz)
        return np.stack([cy, cz], axis=-1)
    pieces = []
    g = model.gamma
    for block, A, B, Q, R in [(0, model.A, model.B, model.Q, model.R), (1, model.At, model.Bt, model.Qt, model.Rt)]:
        G = theta[..., block, :, :]
        F = A-B@G
        if np.max(np.abs(np.linalg.eigvals(np.sqrt(g)*F))) >= 1:
            raise ValueError('Discounted unstable matrix gain')
        S = Q+G.swapaxes(-1, -2)@R@G
        operator = np.eye(4)-g*np.einsum('...ai,...bj->...ijab', F, F).reshape(F.shape[:-2]+(4, 4))
        P = np.linalg.solve(operator, S.reshape(S.shape[:-2]+(4, 1))).reshape(S.shape)
        if np.max(np.abs(P-S-g*F.swapaxes(-1, -2)@P@F)) > 2e-12:
            raise ValueError('Lyapunov residual exceeds tolerance')
        pieces.append(np.einsum('...ij,ji->...', P, model.W))
    return np.stack(pieces, axis=-1)


def evaluate(model, theta, dimension, N=None):
    parts = cost_parts(model, theta, dimension)
    star_parts = cost_parts(model, optimum(model, dimension), dimension)
    cost, star = parts.sum(axis=-1), star_parts.sum()
    result = dict(MF_cost=float(cost), MF_relative_error=float((cost-star)/star))
    if N is not None:
        weights = np.array([1-1/N, 1+1/N])
        own, own_star = parts@weights, star_parts@weights
        result.update(own_cost=float(own), own_relative_error=float((own-own_star)/own_star))
    return result
