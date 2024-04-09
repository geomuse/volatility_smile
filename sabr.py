import numpy as np
import QuantLib as ql

# from quant import find_sigma_with_black_model

class sabr_model:

    def calculate_volatility(self,f,K,T,beta,alpha,rho,nu,ref=False):
        if ref : 
            x = np.log(f/K)
            if x == 0 : 
                IB = alpha*K**(beta-1)
            elif nu==0 : 
                IB = (x*alpha*(1-beta))/(f**(1-beta)-K**(1-beta))
            elif beta == 1 : 
                z = nu * x / alpha
                IB = (nu*x)/np.log((np.sqrt(1-2*rho*z+z**2)+z-rho)/(1-rho) )
            elif beta < 1 :
                z = (nu *(f**(1-beta)-K**(1-beta)))/(alpha*(1-beta))
                IB = (nu*x)/np.log((np.sqrt(1-2*rho*z+z**2))/(1-rho))
            else : 
                ...
            IH = 1 + ( (1-beta)**2/24*alpha**2/(f*K)**(1-beta) + .25 *(rho * beta * nu * alpha)/(f*K)**((1-beta)/2) +(2-3*rho**2)/24 * nu**2)*T
            sigma = IB * IH

        else : 
            if np.any(np.abs(f-K) >= 1e-6) :
                z = nu/alpha*(f*K)**((1-beta)/2)*np.log(f/K)
                x = np.log((np.sqrt(1-2*rho*z+z*z)+z-rho)/(1-rho))
                A = alpha/((f*K)**((1-beta)/2)*1+(1-beta)**2/24*np.log(f/K)**2+(1-beta)**4/1920*np.log(f/K))
                B = z/x
                C = 1 + ((1-beta)**2/24*alpha**2/(f*K)**(1-beta)+1/4*(rho*beta*nu*alpha)/(f*K)**((1-beta)/2)+(2-3*rho**2)/24*nu**2)*T
            else :
                A = alpha/(f**(1-beta))
                B = 1
                C = 1 + ((1-beta)**2/24*alpha**2/f**(2-2*beta)+1/4*(rho*beta*nu*alpha)/f**(1-beta)+(2-3*rho**2)/24*nu**2)*T
            sigma = A*B*C
        return sigma

    def rmse(self,vol,f,K,T,beta):
        return lambda alpha,rho,nu : np.sqrt(np.array([np.power(vol[i] - self.calculate_volatility(f,K[i],T,beta,alpha,rho,nu),2) for i in range(5)]).sum())

    def optimization(self,vol,f,K,T,beta):
        init = ql.Array(3)
        init[0] = 0.01
        init[1] = -0.001
        init[2] = 0.01

        maxIterations = 10000
        minStatIterations = 9999 
        rootEpsilon = 1e-16
        functionEpsilon = 1e-16
        gradientNormEpsilon = 1e-16

        myEndCrit = ql.EndCriteria(maxIterations , minStatIterations , rootEpsilon , functionEpsilon ,
        gradientNormEpsilon)

        constraint = ql.NoConstraint()
        er = self.rmse(vol,f,K,T,beta)

        out = ql.Optimizer().solve(function=er,c=constraint,e=myEndCrit,m=ql.Simplex(1.0),iv=init)
        alpha,rho,nu = np.array(out)
        return alpha,rho,nu