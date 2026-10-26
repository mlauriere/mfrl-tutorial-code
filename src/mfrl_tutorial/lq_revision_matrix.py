"""Checked explicit2D NumPy rollout and raw one-point estimator."""
import numpy as np
from .lq_revision_matrix_reference import Model, inputs

def rollout(model, policies, draws, method):
    """Same full matrix model, with explicit 2D contractions and agent-major states."""
    K, L = policies[:, 0], policies[:, 1]
    FK, FL = model.A-model.B@K, model.At-model.Bt@L
    SK = model.Q+np.swapaxes(K, 1, 2)@(model.R@K)
    SL = model.Qt+np.swapaxes(L, 1, 2)@(model.Rt@L)
    total = np.zeros(len(policies))
    if method == 'MKV':
        y0, y1 = draws['private0'][:, 0, 0].copy(), draws['private0'][:, 0, 1].copy()
        z0, z1 = draws['common0'][:, 0].copy(), draws['common0'][:, 1].copy()
        for t in range(len(draws['common_noise'])):
            total += model.gamma**t*(SK[:,0,0]*y0*y0+(SK[:,0,1]+SK[:,1,0])*y0*y1+SK[:,1,1]*y1*y1
                +SL[:,0,0]*z0*z0+(SL[:,0,1]+SL[:,1,0])*z0*z1+SL[:,1,1]*z1*z1)
            ny0 = FK[:,0,0]*y0+FK[:,0,1]*y1+draws['private_noise'][t,:,0,0]
            ny1 = FK[:,1,0]*y0+FK[:,1,1]*y1+draws['private_noise'][t,:,0,1]
            nz0 = FL[:,0,0]*z0+FL[:,0,1]*z1+draws['common_noise'][t,:,0]
            nz1 = FL[:,1,0]*z0+FL[:,1,1]*z1+draws['common_noise'][t,:,1]
            y0, y1, z0, z1 = ny0, ny1, nz0, nz1
    else:
        x0 = draws['private0'][:,:,0].T.copy()+draws['common0'][:,0]
        x1 = draws['private0'][:,:,1].T.copy()+draws['common0'][:,1]
        for t in range(len(draws['common_noise'])):
            z0, z1 = x0.mean(axis=0), x1.mean(axis=0)
            y0, y1 = x0-z0, x1-z1
            total += model.gamma**t*((SK[:,0,0]*y0*y0+(SK[:,0,1]+SK[:,1,0])*y0*y1+SK[:,1,1]*y1*y1).mean(axis=0)
                +SL[:,0,0]*z0*z0+(SL[:,0,1]+SL[:,1,0])*z0*z1+SL[:,1,1]*z1*z1)
            nx0 = (FK[:,0,0]*y0+FK[:,0,1]*y1+FL[:,0,0]*z0+FL[:,0,1]*z1
                   +draws['private_noise'][t,:,:,0].T+draws['common_noise'][t,:,0])
            nx1 = (FK[:,1,0]*y0+FK[:,1,1]*y1+FL[:,1,0]*z0+FL[:,1,1]*z1
                   +draws['private_noise'][t,:,:,1].T+draws['common_noise'][t,:,1])
            x0, x1 = nx0, nx1
    return total

def gradient_from_samples(costs, directions, tau, estimator):
    if estimator == 'raw':
        weights, denominator = costs, len(costs)
    elif estimator == 'loo_baseline':
        # Each leave-one-out baseline is independent of its own direction/cost.
        # M/(M-1) corrects the bias of naive within-batch centering.
        weights, denominator = costs-costs.mean(), len(costs)-1
    else:
        raise ValueError(estimator)
    return 4*np.einsum('m,mqij->qij', weights, directions)/(denominator*tau**2)

def estimate(model, theta, config, M, method, step, estimator='raw', diagnostics=False):
    N = 1 if method=='MKV' else int(method.removeprefix('Pop_N'))
    directions, draws = inputs(config, M, N, step)
    costs = rollout(model, theta[None,...]+directions, draws, method)
    gradient = gradient_from_samples(costs, directions, config['tau'], estimator)
    if diagnostics:
        return gradient, dict(mean_sample_cost=float(costs.mean()), sd_sample_cost=float(costs.std(ddof=1)))
    return gradient

def metrics(model, theta, N=None):
    parts=np.array([model.block(theta[0],model.A,model.B,model.Q,model.R)[0],
                    model.block(theta[1],model.At,model.Bt,model.Qt,model.Rt)[0]])
    optimal_parts=np.array([model.block(model.optimum[0],model.A,model.B,model.Q,model.R)[0],
                            model.block(model.optimum[1],model.At,model.Bt,model.Qt,model.Rt)[0]])
    cost=float(parts.sum()); delta=theta-model.optimum
    norms=np.linalg.norm(delta,axis=(1,2))
    answer=dict(theta=theta.tolist(),MF_cost=cost,MF_relative_error=(cost-model.cstar)/model.cstar,
        K_Frobenius_error=float(norms[0]),L_Frobenius_error=float(norms[1]),
        K_relative_error=float(norms[0]/np.linalg.norm(model.optimum[0])),
        L_relative_error=float(norms[1]/np.linalg.norm(model.optimum[1])),
        joint_relative_error=float(np.linalg.norm(delta)/np.linalg.norm(model.optimum)))
    if N is not None:
        weights=np.array([1-1/N,1+1/N]); own=float(weights@parts); star=float(weights@optimal_parts)
        answer.update(own_cost=own,own_relative_error=(own-star)/star)
    return answer
