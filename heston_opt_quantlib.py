import os , sys
import numpy as np
import QuantLib as ql
current_dir = os.path.dirname(os.path.abspath(__file__))
path = '/home/geo/Downloads/geo'
sys.path.append(path)

from financial_bot.volatility_smile.heston import heston_model
from financial_bot.options.options import black_scholes , european_option_heston_model
    
if __name__ == '__main__' :

    strike_price = 370
    rd = 3.8580/100
    rf = 0.0
    step = 50
    epoch = 250000
    so = 355.5
    T = 2.0

    vo = 0.04 
    kappa = 2.0
    theta = 0.04
    sigma = 0.1
    rho = -0.7
    lambdav = 0.0
    nu0 = 1.0

    price = european_option_heston_model().european_option_heston(so, strike_price, T, rd, vo, kappa, theta, sigma, rho)
    bs_call = black_scholes().call(so,strike_price,T,rd,rf,sigma)
    print(f"Estimated European Call Option Price under Heston Model Using Monte Carlo : {price}")
    print(f"Estimated European Call Option Price under Heston Model Using Formula : {bs_call}")

    def error(x):
        vo , kappa , theta , sigma , rho = x
        price = european_option_heston_model().european_option_heston(so,strike_price,T,rd,vo,kappa,theta,sigma,rho)
        bs_call = black_scholes().call(so,strike_price,T,rd,rf,sigma)
        return (price-bs_call)**2
    
    init = ql.Array(5)
    init[0] = 0.04
    init[1] = 2.0
    init[2] = 0.04
    init[3] = 0.1
    init[4] = -0.7

    maxIterations = 10000
    minStatIterations = 9999 
    rootEpsilon = 1e-16
    functionEpsilon = 1e-16
    gradientNormEpsilon = 1e-16

    myEndCrit = ql.EndCriteria(maxIterations , minStatIterations , rootEpsilon , functionEpsilon ,
    gradientNormEpsilon)

    constraint = ql.NoConstraint()

    out = ql.Optimizer().solve(function=error,c=constraint,e=myEndCrit,m=ql.Simplex(1.0),iv=init)
    vo , kappa , theta , sigma , rho = np.array(out)