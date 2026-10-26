"""Frozen scalar batched estimator, extracted from the checked research implementation."""
import numpy as np
SIGNS = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])

def rng(seed, stage, step, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, stage, step, stream]))

def primitives(model, count, T, N, seed, stage, step):
    """Private agent-prefix coupling across N, and common streams across oracles.

    At equal count/T, Pop2 is the first two private streams of Pop100, and MKV
    uses the first private stream. Different counts are NOT claimed coupled.
    Innovations include the last unused draw, as in the historical simulator.
    """
    return dict(
        private0=rng(seed, stage, step, 1).uniform(-1, 1, (N, count)).T,
        common0=rng(seed, stage, step, 3).uniform(-1, 1, count),
        private_noise=rng(seed, stage, step, 2).normal(
            0, np.sqrt(model.var_eps), (N, T, count)).transpose(1, 2, 0),
        common_noise=rng(seed, stage, step, 4).normal(
            0, np.sqrt(model.var_eps0), (T, count)))

def directions(p, seed, stage, step):
    calls = p['cost_calls_per_update']
    count = calls // 2 if p['estimator'] == 'two_point_shared_noise' else calls
    if p['estimator'] == 'one_point_balanced_four_signs':
        if count % 4:
            raise ValueError('Balanced directions require a multiple of four')
        return np.tile(SIGNS, (count // 4, 1)) * p['tau']
    return rng(seed, stage, step, 0).choice([-1., 1.], (count, 2)) * p['tau']

def rollout(model, policies, draws, method):
    """Evaluate heterogeneous per-row policies on explicitly supplied primitives."""
    K, L = np.asarray(policies).T
    common0, en0 = draws['common0'], draws['common_noise']
    total = np.zeros(len(K))
    if method == 'MKV':
        y, z = draws['private0'][:, 0].copy(), common0.copy()
        for t in range(len(en0)):
            total += model.gamma**t * ((model.q+model.r*K*K)*y*y
                                      +(model.q+model.q_bar+(model.r+model.r_bar)*L*L)*z*z)
            y = (model.A-model.B*K)*y + draws['private_noise'][t, :, 0]
            z = (model.A_tilde-model.B_tilde*L)*z + en0[t]
    else:
        x = draws['private0'] + common0[:, None]
        for t in range(len(en0)):
            xb = x.mean(axis=1, keepdims=True)
            u = -K[:, None]*(x-xb)-L[:, None]*xb
            ub = u.mean(axis=1, keepdims=True)
            total += model.gamma**t * (model.q*x*x+model.q_bar*xb*xb
                                      +model.r*u*u+model.r_bar*ub*ub).mean(axis=1)
            x = (model.a*x+model.a_bar*xb+model.b*u+model.b_bar*ub
                 +draws['private_noise'][t]+en0[t, :, None])
    return total

def estimate(model, theta, p, method, seed, stage, step, return_inputs=False):
    v = directions(p, seed, stage, step)
    N = 1 if method == 'MKV' else int(method.split('N')[-1])
    draws = primitives(model, len(v), p['T'], N, seed, stage, step)
    plus = rollout(model, theta+v, draws, method)
    if p['estimator'] == 'two_point_shared_noise':
        minus = rollout(model, theta-v, draws, method)
        weights = (plus-minus)/2
    else:
        minus = None
        weights = plus
    gradient = np.mean(weights[:, None]*v, axis=0)/p['tau']**2
    if return_inputs:
        return gradient, (v, draws, plus, minus)
    return gradient

def factors(model, theta, tau=0.):
    K, L = theta
    return np.array([model.gamma*(abs(model.A-model.B*K)+abs(model.B)*tau)**2,
                     model.gamma*(abs(model.A_tilde-model.B_tilde*L)+abs(model.B_tilde)*tau)**2])

def cost_components(model, theta):
    K, L = theta
    rho = factors(model, theta)
    if np.max(rho) >= 1 or not np.all(np.isfinite(theta)):
        raise ValueError(f'Inadmissible base policy {theta}, factors {rho}')
    return np.array([(model.Q+model.R*K*K)*(model.var_y0+model.gamma*model.var_eps/(1-model.gamma))/(1-rho[0]),
                     (model.Q_tilde+model.R_tilde*L*L)*(model.var_z0+model.gamma*model.var_eps0/(1-model.gamma))/(1-rho[1])])
