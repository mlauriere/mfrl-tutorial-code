"""Optional fused C rollout and reusable fresh-noise buffer."""
import ctypes
from pathlib import Path
import numpy as np
from . import lq_revision_matrix as core
from .lq_revision_matrix_reference import rng
HERE=Path(__file__).resolve().parent
_lib=None

def inputs_standard(config,M,N,step):
    """Same generator draws/order; normal(0,sigma) split into fill and scaling."""
    seed,T,tau=config['seed'],config['T'],config['tau'];sd=np.sqrt(config['innovation_covariance_per_component'])
    v=rng(seed,M,step,0).normal(size=(M,2,2,2));v*=tau/np.linalg.norm(v,axis=(2,3),keepdims=True)
    p0=rng(seed,M,step,1).uniform(-1,1,(N,M,2)).transpose(1,0,2)
    c0=rng(seed,M,step,2).uniform(-1,1,(M,2))
    pn=np.empty((N,T,M,2));rng(seed,M,step,3).standard_normal(out=pn);pn*=sd
    cn=np.empty((T,M,2));rng(seed,M,step,4).standard_normal(out=cn);cn*=sd
    return v,dict(private0=p0,common0=c0,private_noise=pn.transpose(1,2,0,3),common_noise=cn)

def rollout_fused(model,policies,draws,method):
    global _lib
    if method=='MKV':return core.rollout(model,policies,draws,method)
    if _lib is None:
        _lib=ctypes.CDLL(str(native_library_path()))
        _lib.rollout_population.argtypes=[ctypes.c_size_t]*3+[ctypes.c_double]+[ctypes.c_void_p]*9
        _lib.rollout_population.restype=ctypes.c_int
    K,L=policies[:,0],policies[:,1]
    mats=[model.A-model.B@K,model.At-model.Bt@L,model.Q+np.swapaxes(K,1,2)@(model.R@K),model.Qt+np.swapaxes(L,1,2)@(model.Rt@L)]
    p0=draws['private0'].transpose(1,0,2);pn=draws['private_noise'].transpose(2,0,1,3)
    arrays=[np.ascontiguousarray(x) for x in mats+[p0,draws['common0'],pn,draws['common_noise']]]
    out=np.empty(len(policies));arrays.append(out)
    rc=_lib.rollout_population(len(policies),p0.shape[0],pn.shape[1],model.gamma,*[x.ctypes.data for x in arrays])
    if rc:raise MemoryError('C rollout allocation failed')
    return out

class InputBuffer:
    """Reuse full-horizon noise storage; every entry is freshly refilled each step.

    No RNG state is retained: seed/M/step/stream still fully determines draws.
    Holding this buffer does not reuse randomness across training updates.
    """
    def __init__(self,config,M,N):
        self.M,self.N,self.T=M,N,config['T']
        self.pn=np.empty((N,self.T,M,2));self.cn=np.empty((self.T,M,2))
    def __call__(self,config,M,N,step):
        assert (M,N,config['T'])==(self.M,self.N,self.T)
        seed,tau=config['seed'],config['tau'];sd=np.sqrt(config['innovation_covariance_per_component'])
        v=rng(seed,M,step,0).normal(size=(M,2,2,2));v*=tau/np.linalg.norm(v,axis=(2,3),keepdims=True)
        p0=rng(seed,M,step,1).uniform(-1,1,(N,M,2)).transpose(1,0,2)
        c0=rng(seed,M,step,2).uniform(-1,1,(M,2))
        rng(seed,M,step,3).standard_normal(out=self.pn);self.pn*=sd
        rng(seed,M,step,4).standard_normal(out=self.cn);self.cn*=sd
        return v,dict(private0=p0,common0=c0,private_noise=self.pn.transpose(1,2,0,3),common_noise=self.cn)

def estimate(model,theta,config,M,method,step,variant='standard',return_samples=False,input_buffer=None):
    N=1 if method=='MKV' else int(method.removeprefix('Pop_N'))
    v,draws=(input_buffer or inputs_standard)(config,M,N,step)
    if variant=='fused':costs=rollout_fused(model,theta[None]+v,draws,method)
    elif variant=='standard':costs=core.rollout(model,theta[None]+v,draws,method)
    else:raise ValueError(variant)
    grad=core.gradient_from_samples(costs,v,config['tau'],'raw')
    return (grad,costs,v) if return_samples else grad

def native_library_path():
    import sys
    suffix=".dylib" if sys.platform=="darwin" else ".so"
    return HERE/"native"/("rollout_fused"+suffix)
