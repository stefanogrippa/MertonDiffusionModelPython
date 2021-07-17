# https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

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





S = 100 # current stock price
T = 1  # time to maturity
R = 0.02  # risk free rate
M = 0  # mean of jump size
V = 0.3  # standard deviation of jump
Lam = 1  # intensity of jump i.e. number of jumps per year
Steps = 10000  # time steps
Npaths = 2  # number of paths to simulate
Sigma = 0.2  # annual standard deviation , for Weiner process

j = merton_jump_paths(S, T, R, Sigma, Lam, M, V, Steps, Npaths)

plt.plot(j)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Jump Diffusion Process')
plt.show()


# closed form solution
N = norm.cdf


def bs_call(s, k, t, r, sigma):
    d1 = (np.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * N(d1) - k * np.exp(-r * t) * N(d2)


def bs_put(s, k, t, r, sigma):
    d1 = (np.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return k * np.exp(-r * t) * N(-d2) - s * N(-d1)


def merton_jump_call(s, k, t, r, sigma, m, v, lam):
    p = 0
    for k in range(40):
        r_k = r - lam * (m - 1) + (k * np.log(m)) / T
        sigma_k = np.sqrt(sigma ** 2 + (k * v ** 2) / T)
        k_fact = np.math.factorial(k)
        p += (np.exp(-m * lam * T) * (m * lam * T) ** k / k_fact) * bs_call(s, k, t, r_k, sigma_k)

    return p


def merton_jump_put(s, k, t, r, sigma, m, v, lam):
    p = 0  # price of option
    for k in range(40):
        r_k = r - lam * (m - 1) + (k * np.log(m)) / T
        sigma_k = np.sqrt(sigma ** 2 + (k * v ** 2) / T)
        k_fact = np.math.factorial(k)  #
        p += (np.exp(-m * lam * T) * (m * lam * T) ** k / k_fact) \
             * bs_put(s, k, t, r_k, sigma_k)
    return p

# https://www.codearmo.com/python-tutorial/calculating-volatility-smile
def implied_vol(opt_value, s, k, t, r, type_='call'):
    def call_obj(sigma):
        return abs(bs_call(s, k, t, r, sigma) - opt_value)

    def put_obj(sigma):
        return abs(bs_put(s, k, t, r, sigma) - opt_value)

    if type_ == 'call':
        res = minimize_scalar(call_obj, bounds=(0.01, 6), method='bounded')
        return res.x
    elif type_ == 'put':
        res = minimize_scalar(put_obj, bounds=(0.01, 6),
                              method='bounded')
        return res.x
    else:
        raise ValueError("type_ must be 'put' or 'call'")


K = 100
np.random.seed(3)
j = merton_jump_paths(S, T, R, Sigma, Lam, M, V, Steps, Npaths) #generate jump diffusion paths

mcprice = np.maximum(j[-1]-K,0).mean() * np.exp(-R*T) # calculate value of call

cf_price = merton_jump_call(S, K, T, R, Sigma, np.exp(M+V**2*0.5), V, Lam)

print('Merton Price =', cf_price)
print('Monte Carlo Merton Price =', mcprice)
print('Black Scholes Price =', bs_call(S,K,T,R, Sigma))

#Merton Price = 14.500570058304778
#Monte Carlo Merton Price = 14.597509592911369
#Black Scholes Price = 8.916037278572539

strikes = np.arange(50,150,1)

mjd_prices = merton_jump_call(S, strikes, T, R, Sigma, M, V, Lam)
merton_ivs = [implied_vol(c, S, k, T, R) for c,k in zip(mjd_prices, strikes)]

plt.plot(strikes, merton_ivs, label='IV Smile')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.axvline(S, color='black', linestyle='dashed', linewidth=2, label="Spot")
plt.title('MJD Volatility Smile')
plt.legend()
plt.show()
