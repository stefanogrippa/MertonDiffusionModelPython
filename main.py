# https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python

# modello di Merton con jumps a volatilita costante

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import pandas as pd
import seaborn as sns
# import time
from scipy.optimize import minimize

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


S = 100  # current stock price
T = 1  # time to maturity
R = 0.02  # risk free rate
M = 0  # mean of jump size
V = 0.3  # standard deviation of jump
Lam = 1  # intensity of jump i.e. number of jumps per year
Steps = 10000  # time steps
Npaths = 2  # number of paths to simulate
Sigma = 0.2  # annual standard deviation , for Weiner process

K = 100  # strike price (l'opzione in questo momento e' at the money)

j = merton_jump_paths(S, T, R, Sigma, Lam, M, V, Steps, Npaths)

plt.plot(j)
plt.xlabel('Days')
plt.ylabel('Stock Price')
Plottitle = 'Jump Diffusion Process for ' + str(Npaths) + ' paths'
plt.title(Plottitle)
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


def merton_jump_call(s, kappa, t, r, sigma, m, v, lam):
    p = 0
    for k in range(40):
        r_k = r - lam * (m - 1) + (k * np.log(m)) / t
        sigma_k = np.sqrt(sigma ** 2 + (k * v ** 2) / t)
        k_fact = np.math.factorial(k)
        p += (np.exp(-m * lam * t) * (m * lam * t) ** k / k_fact) * bs_call(s, kappa, t, r_k, sigma_k)

    return p


def merton_jump_put(s, kappa, t, r, sigma, m, v, lam):
    p = 0  # price of option
    for k in range(40):
        r_k = r - lam * (m - 1) + (k * np.log(m)) / T
        sigma_k = np.sqrt(sigma ** 2 + (k * v ** 2) / T)
        k_fact = np.math.factorial(k)  #
        p += (np.exp(-m * lam * T) * (m * lam * T) ** k / k_fact) * bs_put(s, kappa, t, r_k, sigma_k)
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



np.random.seed(3)
j = merton_jump_paths(S, T, R, Sigma, Lam, M, V, Steps, Npaths)  # generate jump diffusion paths

mcprice = np.maximum(j[-1]-K,0).mean() * np.exp(-R*T)  # calculate value of call


cf_price = merton_jump_call(S, K, T, R, Sigma, np.exp(M+V**2*0.5), V, Lam)

# print('Merton Price =', cf_price)
print('Monte Carlo Merton Price =', mcprice)
print('Black Scholes Price =', bs_call(S,K,T,R, Sigma))

#Merton Price = 14.500570058304778
#Monte Carlo Merton Price = 14.597509592911369
#Black Scholes Price = 8.916037278572539

strikes = np.arange(50, 150, 1)


mjd_prices = merton_jump_call(S, strikes, T, R, Sigma, M, V, Lam)
merton_ivs = [implied_vol(c, S, k, T, R) for c,k in zip(mjd_prices, strikes)]

plt.plot(strikes, merton_ivs, label='IV Smile')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.axvline(S, color='black', linestyle='dashed', linewidth=2, label="Spot")
plt.title('MJD Volatility Smile')


plt.legend()
plt.show()

# calibration
df = pd.read_csv('https://raw.githubusercontent.com/codearmo/data/master/calls_calib_example.csv')

print(df.head(10))

def optimal_params(x, mkt_prices, strikes):
    candidate_prices = merton_jump_call(S, strikes, T, R,
                                        sigma=x[0], m= x[1] ,
                                        v=x[2],lam= x[3])
    return np.linalg.norm(mkt_prices - candidate_prices, 2)


T = df['T'].values[0]
S = df.F.values[0]
R = 0
x0 = [0.15, 1, 0.1, 1]  # initial guess for algorithm
bounds = ((0.01, np.inf), (0.01, 2), (1e-5, np.inf) , (0, 5))  #bounds as described above
strikes = df.Strike.values
prices = df.Midpoint.values
# minimize non definito
Res = minimize(optimal_params, method='SLSQP',  x0=x0, args=(prices, strikes), bounds = bounds, tol=1e-20, options={"maxiter":1000})
sigt = Res.x[0]
mt = Res.x[1]
vt = Res.x[2]
lamt = Res.x[3]

print('Calibrated Volatlity = ', sigt)
print('Calibrated Jump Mean = ', mt)
print('Calibrated Jump Std = ', vt)
print('Calibrated intensity = ', lamt)

#Calibrated Volatlity =  0.06489478237064618
#Calibrated Jump Mean =  0.8789051095314648
#Calibrated Jump Std =  0.1542041201811455
#Calibrated intensity =  0.9722952134238365

df['least_sq_V'] = merton_jump_call(S, df.Strike, df['T'], 0, sigt, mt, vt, lamt)

plt.scatter(df.Strike, df.Midpoint,label= 'Observed Prices')
plt.plot(df.Strike, df.least_sq_V, color='black',label= 'Fitted Prices')
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Value in $')
plt.title('Merton Model Optimal Params')
plt.show()

# https://www.codearmo.com/python-tutorial/heston-model-simulation-python
rho = -0.7
Ndraws = 1000
mu = np.array([0, 0])
cov = np.array([[1, rho], [rho , 1]])

W = np.random.multivariate_normal(mu, cov, size=Ndraws)

plt.plot(W.cumsum(axis=0))
plt.title('Correlated Random Variables')
plt.show()
print(np.corrcoef(W.T))


def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi,
                          steps, Npaths, return_vol=False):
    dt = T / steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0, 0]),
                                           cov=np.array([[1, rho],
                                                         [rho, 1]]),
                                           size=paths) * np.sqrt(dt)

        S_t = S_t * (np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[:, 0]))
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = S_t
        sigs[:, t] = v_t

    if return_vol:
        return prices, sigs

    return prices

kappa =4
theta = 0.02
v_0 =  0.02
xi = 0.9
r = 0.02
S = 100
paths =50000
steps = 2000
T = 1

prices_pos = generate_heston_paths(S, T, r, kappa, theta,
                                    v_0, rho=0.9, xi=xi, steps=steps, Npaths=paths,
                                    return_vol=False)[:,-1]
prices_neg  = generate_heston_paths(S, T, r, kappa, theta,
                                    v_0, rho=-0.9, xi=xi, steps=steps, Npaths=paths,
                                    return_vol=False)[:,-1]
gbm_bench = S*np.exp( np.random.normal((r - v_0/2)*T ,
                                np.sqrt(theta)*np.sqrt(T), size=paths))



fig, ax = plt.subplots()

ax = sns.kdeplot(data=prices_pos, label=r"$\rho = 0.9$", ax=ax)
ax = sns.kdeplot(data=prices_neg, label=r"$\rho= -0.9$ ", ax=ax)
ax = sns.kdeplot(data=gbm_bench, label="GBM", ax=ax)

ax.set_title(r'Tail Density by Varying $\rho$')
plt.axis([40, 180, 0, 0.055])
plt.xlabel('$S_T$')
plt.ylabel('Density')
plt.show()

strikes =np.arange(30, 200,1)

puts = []

for K in strikes:
    P = np.mean(np.maximum(K-prices_neg,0))*np.exp(-r*T)
    puts.append(P)


ivs = [implied_vol(P, S, K, T, r, type_ = 'put' ) for P, K in zip(puts,strikes)]

plt.plot(strikes, ivs)
plt.ylabel('Implied Volatility')
plt.xlabel('Strike')
plt.axvline(S, color='black',linestyle='--',
            label='Spot Price')
plt.title('Implied Volatility Smile from Heston Model')
plt.legend()
plt.show()

kappa = 3
theta = 0.04
v_0 = 0.04
xi = 0.6
r = 0.05
S = 100       
paths = 3
steps = 10000
T = 1
rho = -0.8
prices, sigs = generate_heston_paths(S, T, r, kappa, theta,
                                     v_0, rho, xi, steps, paths,
                                     return_vol=True)

plt.figure(figsize=(7, 6))
plt.plot(prices.T)
plt.title('Heston Price Paths Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.show()

plt.figure(figsize=(7, 6))
plt.plot(np.sqrt(sigs).T)
plt.axhline(np.sqrt(theta), color='black', label=r'$\sqrt{\theta}$')
plt.title('Heston Stochastic Vol Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Volatility')
plt.legend(fontsize=15)
plt.show()