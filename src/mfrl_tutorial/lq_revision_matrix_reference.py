"""Dense matrix model and original indexed inputs; no job or file side effects."""
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

class Model:
    def __init__(self, config):
        self.gamma = float(config['gamma'])
        for name in ['A', 'A_bar', 'B', 'B_bar', 'Q', 'Q_bar', 'R', 'R_bar']:
            setattr(self, name, np.asarray(config[name], dtype=float))
        self.At, self.Bt = self.A+self.A_bar, self.B+self.B_bar
        self.Qt, self.Rt = self.Q+self.Q_bar, self.R+self.R_bar
        self.S0 = np.eye(2)*config['initial_covariance_per_component']
        self.Sn = np.eye(2)*config['innovation_covariance_per_component']
        self.W = self.S0+self.gamma/(1-self.gamma)*self.Sn
        self.optimum = np.array([self.riccati(self.A, self.B, self.Q, self.R),
                                 self.riccati(self.At, self.Bt, self.Qt, self.Rt)])
        self.cstar = self.cost(self.optimum)

    def riccati(self, A, B, Q, R):
        P = solve_discrete_are(np.sqrt(self.gamma)*A, np.sqrt(self.gamma)*B, Q, R)
        return np.linalg.solve(R+self.gamma*B.T@P@B, self.gamma*B.T@P@A)

    def block(self, G, A, B, Q, R):
        F = A-B@G
        if max(abs(np.linalg.eigvals(np.sqrt(self.gamma)*F))) >= 1:
            raise ValueError('Discounted unstable gain')
        P = solve_discrete_lyapunov(np.sqrt(self.gamma)*F.T, Q+G.T@R@G)
        Sigma = solve_discrete_lyapunov(np.sqrt(self.gamma)*F, self.W)
        grad = 2*((R+self.gamma*B.T@P@B)@G-self.gamma*B.T@P@A)@Sigma
        return float(np.trace(P@self.W)), grad, P

    def cost(self, theta, N=None, gradient=False):
        cy, gy, _ = self.block(theta[0], self.A, self.B, self.Q, self.R)
        cz, gz, _ = self.block(theta[1], self.At, self.Bt, self.Qt, self.Rt)
        wy, wz = (1., 1.) if N is None else (1-1/N, 1+1/N)
        return (wy*cy+wz*cz, np.array([wy*gy, wz*gz])) if gradient else wy*cy+wz*cz

    def factors(self, theta, tau=0.):
        return np.array([self.gamma*(np.linalg.norm(A-B@G, 2)+np.linalg.norm(B, 2)*tau)**2
                         for G, A, B in [(theta[0], self.A, self.B), (theta[1], self.At, self.Bt)]])

def rng(seed, M, step, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, M, step, stream]))

def inputs(config, M, N, step):
    seed, T, tau = config['seed'], config['T'], config['tau']
    noise_sd = np.sqrt(config['innovation_covariance_per_component'])
    v = rng(seed, M, step, 0).normal(size=(M, 2, 2, 2))
    v *= tau/np.linalg.norm(v, axis=(2, 3), keepdims=True)
    # N is the leading RNG dimension: smaller N uses the same private prefixes.
    draws = dict(private0=rng(seed, M, step, 1).uniform(-1, 1, (N, M, 2)).transpose(1, 0, 2),
        common0=rng(seed, M, step, 2).uniform(-1, 1, (M, 2)),
        private_noise=rng(seed, M, step, 3).normal(0, noise_sd, (N, T, M, 2)).transpose(1, 2, 0, 3),
        common_noise=rng(seed, M, step, 4).normal(0, noise_sd, (T, M, 2)))
    return v, draws

def rollout(model, policies, draws, method):
    K, L = policies[:, 0], policies[:, 1]
    FK, FL = model.A-model.B@K, model.At-model.Bt@L
    SK = model.Q+np.einsum('mai,ab,mbj->mij', K, model.R, K)
    SL = model.Qt+np.einsum('mai,ab,mbj->mij', L, model.Rt, L)
    total = np.zeros(len(policies))
    if method == 'MKV':
        y, z = draws['private0'][:, 0].copy(), draws['common0'].copy()
        for t in range(len(draws['common_noise'])):
            total += model.gamma**t*(np.einsum('mi,mij,mj->m', y, SK, y)
                                     +np.einsum('mi,mij,mj->m', z, SL, z))
            y = np.einsum('mij,mj->mi', FK, y)+draws['private_noise'][t, :, 0]
            z = np.einsum('mij,mj->mi', FL, z)+draws['common_noise'][t]
    else:
        x = draws['private0']+draws['common0'][:, None, :]
        for t in range(len(draws['common_noise'])):
            xb = x.mean(axis=1)
            y = x-xb[:, None, :]
            total += model.gamma**t*(np.einsum('mni,mij,mnj->mn', y, SK, y).mean(axis=1)
                                     +np.einsum('mi,mij,mj->m', xb, SL, xb))
            x = (np.einsum('mij,mnj->mni', FK, y)
                 +np.einsum('mij,mj->mi', FL, xb)[:, None, :]
                 +draws['private_noise'][t]+draws['common_noise'][t, :, None, :])
    return total

def direct_rollout(model, policies, draws, method):
    """Unbatched original state/action equations for orientation checks."""
    costs = []
    for row, (K, L) in enumerate(policies):
        total = 0.
        if method == 'MKV':
            y, z = draws['private0'][row, 0].copy(), draws['common0'][row].copy()
            for t in range(len(draws['common_noise'])):
                uy, uz = -K@y, -L@z
                total += model.gamma**t*(y@model.Q@y+z@model.Qt@z+uy@model.R@uy+uz@model.Rt@uz)
                y = model.A@y+model.B@uy+draws['private_noise'][t, row, 0]
                z = model.At@z+model.Bt@uz+draws['common_noise'][t, row]
        else:
            x = draws['private0'][row]+draws['common0'][row]
            for t in range(len(draws['common_noise'])):
                xb = x.mean(axis=0)
                u = np.array([-K@(xx-xb)-L@xb for xx in x])
                ub = u.mean(axis=0)
                total += model.gamma**t*np.mean([(xx-xb)@model.Q@(xx-xb)+xb@model.Qt@xb
                    +(uu-ub)@model.R@(uu-ub)+ub@model.Rt@ub for xx, uu in zip(x, u)])
                x = np.array([model.A@xx+model.A_bar@xb+model.B@uu+model.B_bar@ub
                    +draws['private_noise'][t, row, j]+draws['common_noise'][t, row]
                    for j, (xx, uu) in enumerate(zip(x, u))])
        costs.append(total)
    return np.asarray(costs)
