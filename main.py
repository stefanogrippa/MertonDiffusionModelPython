# https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


def merton_jump_paths(s, t, r, sigma, lam, m, v, steps, nofpaths):
    size = (steps, nofpaths)
    dt = t / steps
    poi_rv = np.multiply(np.random.poisson(lam * dt, size=size),
                         np.random.normal(m, v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 * 0.5)) * dt +
                     sigma * np.sqrt(dt) *
                     np.random.normal(size=size)), axis=0)

    return np.exp(geo + poi_rv) * s


S = 80  # 100  # current stock price
T = 1  # time to maturity
R = 0.03  # 0.02  # risk free rate
M = 0.02  # 0  # mean of jump size
V = 0.2  # 0.3  # standard deviation of jump
Lam = 0.1  # 1  # intensity of jump i.e. number of jumps per year
Steps = 10000  # time steps
Npaths = 10  # number of paths to simulate
Sigma = 0.2  # annual standard deviation , for Weiner process

j = merton_jump_paths(S, T, R, Sigma, Lam, M, V, Steps, Npaths)

plt.plot(j)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Jump Diffusion Process')
plt.show()
