import os , sys
import numpy as np
from scipy.optimize import minimize
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
    
    init_params = np.array([vo,kappa,theta,sigma,rho])
    result = minimize(error,init_params,method='Nelder-Mead')

    print(result)

    """
    message: Maximum number of function evaluations has been exceeded.
       success: False
        status: 1
           fun: 2.541665069390587e-07
             x: [ 4.436e-02  2.103e+00  1.427e-02  1.444e-01 -5.997e-01]
           nit: 359
          nfev: 1000
        final_simplex: (array([[ 4.436e-02,  2.103e+00, ...,  1.444e-01,
                                -5.997e-01],
                            [ 4.436e-02,  2.103e+00, ...,  1.444e-01,
                                -5.997e-01],
                            ...,
                            [ 4.436e-02,  2.103e+00, ...,  1.444e-01,
                                -5.997e-01],
                            [ 4.436e-02,  2.103e+00, ...,  1.444e-01,
                                -5.997e-01]]), array([ 2.542e-07,  3.255e-02,  6.259e-02,  6.946e-02,
                                2.787e-01,  6.505e-01]))
    """