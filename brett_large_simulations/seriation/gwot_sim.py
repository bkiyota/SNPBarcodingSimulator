import numpy as np
import scipy as sp
import math
import copy


class TimeSeries():
    """Base class for time-series dataset.
    
    :param x: `np.array` of observed datapoints. 
    :param dt: `np.array` of time increments `t[i+1] - t[i]`.
    :param t_idx: `np.array` of time indices for each datapoint in `x`. 
    :param D: diffusivity 
    """
    def __init__(self, x, dt, t_idx, D = None):
        self.x = x
        self.t_idx = t_idx
        self.T = len(np.unique(t_idx))
        self.N = np.unique(t_idx, return_counts = True)[1]
        self.D = D
        self.dt = dt


def dW(dt, sz):
    """ Wiener process increments of size `sz`
    """
    return np.sqrt(dt)*np.random.standard_normal(sz)


def sde_integrate(dV, nu, x0, t, steps, birth_death = False, b = None, d = None, g_max = 50, snaps = None):
    """Integrate SDE using Euler-Maruyama method (with birth-death)
    
    :param dV: function `dV(x, t)` specifying the drift field
    :param nu: diffusivity
    :param x0: initial particle positions at time `t = 0`
    :param steps: time steps to use in Euler-Maruyama method
    :param birth_death: `True` if simulation needs birth-death
    :param b: if `birth_death == True`, birth rate `b(x, t)`
    :param d: if `birth_death == True`, death rate `d(x, t)`
    :param g_max: if `birth_death == True`, we store `g_max*x0.shape[0]` particles 
                and error if exceeded.
    :param snaps: `np.array` of step indices at which to record particle snapshot. 
    """
    if birth_death:
        # store g_max*x0.size[0] particles. Output error if we try and exceed this, though.
        # g = g(x, t) = b(x, t) - d(x, t)
        x = np.zeros((g_max*x0.shape[0], x0.shape[1]))
        x[0:x0.shape[0], :] = x0
    else:
        x = np.array(x0, copy = True)
    
    x_mask = np.zeros(x.shape[0], dtype = bool)
    x_mask[0:x0.shape[0]] = True

    dt = t/steps if steps > 0 else None
    t_current = 0
    snap = np.zeros((len(snaps), ) + x.shape)
    snap_mask = np.zeros((len(snaps), ) + (x.shape[0], ), dtype = bool)
    dV_vec = np.zeros(x.shape)
    
    if steps == 0:
        if snaps is not None and 0 == snaps:
            snap[0] = x
            snap_mask[0] = x_mask

    for i in range(0, steps):
        dV_vec[x_mask, :] = dV(x[x_mask, :], t_current)
        x[x_mask, :] = x[x_mask, :] - dV_vec[x_mask, :]*dt + np.sqrt(nu)*dW(dt, x[x_mask, :].shape)

        # birth/death step
        if birth_death and b is not None and d is not None:
            x_mask_new = x_mask.copy()
            for j in range(0, x.shape[0]):
                if x_mask[j]:
                    u = np.random.uniform()
                    if u < dt*b(x[j, :], t_current):
                        # birth event
                        k = np.where(x_mask_new == False)[0][0]
                        x[k, :] = x[j, :]
                        x_mask_new[k] = True
                    elif u < dt*b(x[j, :], t_current) + dt*d(x[j, :], t_current):
                        # death event
                        x_mask_new[j] = False
                    else:
                        pass
            x_mask = x_mask_new 
        t_current += dt
            
        if snaps is not None and np.sum(i == snaps):
            snap[np.where(i == snaps)[0]] = x
            snap_mask[np.where(i == snaps)[0]] = x_mask
    return snap, snap_mask



class Simulation(TimeSeries):
    """Diffusion-drift SDE simulations using the Euler-Maruyama method. 
    
    :param V: potential function :math:`(x, t) \\mapsto V(x, t)` 
    :param dV: potential gradient :math:`(x, t) \\mapsto \\nabla V(x, t)`
    :param N: number of initial particles to use, :math:`N_i` corresponds to time point :math:`t_i`
    :param T: number of timepoints at which to capture snapshots
    :param d: dimension :math:`d` of simulation
    :param D: diffusivity :math:`D`
    :param t_final: final time :math:`t_\\mathrm{final}` (initial time is always 0)
    :param ic_func: function accepting arguments `(N, d)` and returning an array `X` of dimensions `(N, d)`
                    where `X[i, :]` corresponds to the `i`th initial particle position
    :param pool: ProcessingPool to use for parallel computation (or `None`)
    :param birth_death: whether to incorporate birth-death process
    :param birth: if `birth_death == True`, a function accepting arguments `(X, t)` returning 
                    a vector of birth rates :math:`\\beta` for each row in `X`
    :param death: if `birth_death == True`, a function accepting arguments `(X, t)` returning 
                    a vector of death rates :math:`\\delta` for each row in `X`
    """
    def __init__(self, V, dV, N, T, d, D, t_final, ic_func, pool, birth_death = False, birth = None, death = None):
        self.V = V
        self.dV = dV
        self.birth_death = birth_death
        self.birth = birth
        self.death = death
        self.N = N
        self.d = d
        self.T = T
        self.D = D
        self.t_final = t_final
        self.ic_func = ic_func
        self.dt = (t_final/(T-1))*np.ones(T-1)
        self.pool = pool

    def sample(self, steps_scale = 1, trunc = None):
        """Sample time-series from Simulation. Simulates independent evolving particles using 
            Euler-Maruyama method.

        :param steps_scale: number of Euler-Maruyama steps to take between timepoints. 
        :param trunc: if provided, subsample all snapshots to have `trunc` particles. 
        """
        ic_all = [self.ic_func(self.N[i], self.d) for i in np.arange(0, self.T, 1)]
        def F(i):
            snap, snap_mask = sde_integrate(self.dV, nu = self.D, x0 = ic_all[i],
                                        birth_death = self.birth_death,
                                        b = self.birth, d = self.death, 
                                        t = (self.t_final)*(i/self.T), 
                                        steps = steps_scale*i, 
                                        snaps = np.array([max(steps_scale*i-1, 0), ])) 
            return snap[snap_mask, :]
        if self.pool:
            self.snaps = self.pool.map(F, np.arange(0, self.T, 1))
        else:
            self.snaps = [F(i) for i in np.arange(0, self.T, 1)]

        if trunc is not None:
            samp_sizes = np.array([s.shape[0] for s in self.snaps])
            for i in range(0, len(self.snaps)):
                self.snaps[i] = self.snaps[i][np.random.choice(samp_sizes[i], size = min(samp_sizes[i], trunc)), :]
        self.x = np.vstack(self.snaps) 
        self.t_idx = np.concatenate([np.array([i]).repeat(self.snaps[i].shape[0]) for i in range(0, len(self.snaps))])
        return self.snaps

    def sample_trajectory(self, steps_scale = 1, N = 1):
        """Sample trajectory from simulation

        :param steps_scale: number of Euler-Maruyama steps to take between timepoints. 
        :param N: number of trajectories to sample 
        :return: `np.array` of dimensions 
        """
        ic = self.ic_func(N, self.d)
        snap, snap_mask = sde_integrate(self.dV, nu = self.D, x0 = ic,
            b = self.birth, d = self.death, 
            t = self.t_final, 
            steps = self.T*steps_scale, 
            snaps = np.arange(self.T)*steps_scale) 
        return np.moveaxis(snap, 0, 1)

    def __copy__(self):
        return Simulation(V = self.V, dV = self.dV, N = self.N, T = self.T, d = self.d, D = self.D, t_final = self.t_final,
                         ic_func = self.ic_func, pool = self.pool,
                         birth_death = self.birth_death, birth = self.birth, death = self.death)

    def __deepcopy__(self, memo):
        return Simulation(V = copy.deepcopy(self.V, memo), dV = copy.deepcopy(self.dV, memo), 
                             N = self.N, T = self.T, d = self.d, D = self.D, t_final = self.t_final,
                             ic_func = copy.deepcopy(self.ic_func, memo), pool = self.pool,
                             birth_death = self.birth_death, 
                             birth = copy.deepcopy(self.birth, memo), death = copy.deepcopy(self.death, memo))
