import numpy as np
import pandas as pd
from datetime import datetime
import QuantLib as ql
from scipy import stats
from scipy import integrate

class heston_model:

    # characteristic function
    def cf(self,u,s,kappa,theta,sigma,rho,lambdav,nu0,rd,rf,t,T,typev="both"):
        if typev == "both" :
            zeta = np.array([1,-1])
        else : 
            if typev == 1 : 
                zeta = 1 
            elif typev == 2 :
                zeta = -1
            else : 
                ...
        i = complex(real=0,imag=1)
        x = np.log(s)
        b = kappa + lambdav - (1+zeta)/2 * rho * sigma 
        d = np.sqrt((rho*sigma*i*u-b)**2 - (sigma*sigma) * (zeta*i*u-u*u))
        g = (b-rho*sigma*u*i+d)/(b-rho*sigma*u*i-d)
        if np.real(g) == np.Inf :
            c = (rd-rf)*u*i*(T-t) + kappa * theta / (sigma*sigma)*(b-rho*u*i+d)*(T-t)
        else : 
            c = (rd-rf)*u*i*(T-t) + kappa * theta / (sigma*sigma) *((b-rho*u*i+d)*(T-t)-2*np.log((1-g*np.exp(d*(T-t)))/(1-g)))
        D = (b-rho*sigma*u*i+d)/(sigma * sigma) * ( (1-np.exp(d*(T-t)))/(1-g*np.exp(d*(T-t))) )
        return np.exp(c+D*nu0+i*u*x)

    def heston_init(self,s,K,kappa,theta,sigma,rho,lambdav,nu0,rd,rf,t,T,typev=1):
        '''
        typev == 1 : call option
        typev == -1 : put option
        '''
        i = complex(real=0,imag=1)
        y = np.log(K)
        func_1 = lambda u : np.real(np.exp(-i*u*y)/(i*u)*self.cf(u,s,kappa,theta,sigma,rho,lambdav,nu0,rd,rf,t,T,1))
        func_2 = lambda u : np.real(np.exp(-i*u*y)/(i*u)*self.cf(u,s,kappa,theta,sigma,rho,lambdav,nu0,rd,rf,t,T,2))

        P1 = 1/2 + 1/np.pi * integrate.quad(func_1,1e-10,100)[0]
        P2 = 1/2 + 1/np.pi * integrate.quad(func_2,1e-10,100)[0]

        p1 = (1-typev)/2 + typev * P1
        p2 = (1-typev)/2 + typev * P2

        return typev*(np.exp(-rf*(T-t))*s*p1 - np.exp(-rd*(T-t))*K*p2)

    def heston_call(self,s,K,kappa,theta,sigma,rho,lambdav,nu0,rd,rf,t,T):
        return self.heston_init(s,K,kappa,theta,sigma,rho,lambdav,nu0,rd,rf,t,T,typev=1) 

    def heston_put(self,s,K,kappa,theta,sigma,rho,lambdav,nu0,rd,rf,t,T):
        return self.heston_init(s,K,kappa,theta,sigma,rho,lambdav,nu0,rd,rf,t,T,typev=-1)

    def error(self,theta,sigma,rho,s,T,vol_data,K,rd,rf,kappa,nu0):
        n = len(T)
        price = np.zeros(n*5).reshape(n,5)
        vol = np.zeros(n*5).reshape(n,5)
        
        for i in range(n):
            for j in range(5):
                price[i,j] = self.heston(s,K[i,j],kappa,theta,sigma,rho,0,nu0,rd[i],rf[i],0,T[i])
                vol[i,j] = np.array(self.impvol(price[i,j],s,K[i,j],rd[i],rf[i],0,T[i],1))
                # vol[i,j] = np.array(impvol_na(price[i,j],s,K[i,j],rd[i],rf[i],0,T[i],1,[0.4,0.5,1e-10,500]))
        return np.power(vol - vol_data,2).sum()

    def heston_price_error(self,theta,sigma,rho,s,T,price_data,K,rd,rf,kappa,nu0):
        n = len(T)
        price = [ [ self.heston_call(s,K[i,j],kappa,theta,sigma,rho,0,nu0[i],rd[i],rf[i],0,T[i]) for j in range(5) ] for i in range(n)]
        # return np.power(np.sum(price , axis=1) - np.sum(price_data , axis=1),2).sum()
        return np.power(price - price_data,2).sum()

    def heston_price_error_adjust(self,theta,sigma,rho,s,T,price_data,K,rd,rf,kappa,nu0):
        '''
        nu0 : 为单一变数或在从常数. 
        '''
        n = len(T)
        price = [ [ self.heston_call(s,K[i,j],kappa,theta,sigma,rho,0,nu0,rd[i],rf[i],0,T[i]) for j in range(5) ] for i in range(n)]
        # return np.power(np.sum(price , axis=1) - np.sum(price_data , axis=1),2).sum()
        return np.power(price - price_data,2).sum()

    def heston_optimization(self,s,T,vol_data,price_data,K,rd,rf,nu0,isvol=True):
        init = ql.Array(4)
        init[0], init[1], init[2] , init[3] = 0.04,0.25,0.25,0.5

        lowerBound = ql.Array(4)
        lowerBound[0] = 0.0
        lowerBound[1] = 0.01
        lowerBound[2] = 0.0
        lowerBound[3] = -1.0

        upperBound = ql.Array(4)
        upperBound[0] = 1.0
        upperBound[1] = 15.0
        upperBound[2] = 1.0
        upperBound[3] = 1.0

        maxIterations = 10000
        minStatIterations = 9999 
        rootEpsilon = 1e-10
        functionEpsilon = 1e-10
        gradientNormEpsilon = 1e-10

        myEndCrit = ql.EndCriteria(maxIterations , minStatIterations , rootEpsilon , functionEpsilon ,
        gradientNormEpsilon)

        # constraint = ql.NoConstraint()
        constraint = ql.NonhomogeneousBoundaryConstraint(lowerBound , upperBound)
        if isvol :
            er = lambda theta , kappa , sigma , rho : self.error(theta,sigma,rho,s,T,vol_data,K,rd,rf,kappa,nu0)
        else : 
            er = lambda theta , kappa , sigma , rho : self.heston_price_error_adjust(theta,sigma,rho,s,T,price_data,K,rd,rf,kappa,nu0)
        # method = ql.DifferentialEvolution()
        method = ql.Simplex(1.0)
        out = ql.Optimizer().solve(function=er,c=constraint,e=myEndCrit,m=method,iv=init)
        theta , kappa , sigma , rho = np.array(out)
        if 2*theta * kappa >= (sigma*sigma) :
            '''
            Feller condition
            '''
            print('successfull...')
            if isvol :
                print(self.error(theta,sigma,rho,s,T,vol_data,K,rd,rf,kappa,nu0))
            else :
                print(self.heston_price_error_adjust(theta,sigma,rho,s,T,price_data,K,rd,rf,kappa,nu0))
                
            return theta , kappa , sigma , rho
        else : 
            print('optimzation failed ..')
            return theta , kappa , sigma , rho

    def heston_optimization_5params(self,s,T,vol_data,price_data,K,rd,rf):
        init = ql.Array(5)
        init[0], init[1], init[2] , init[3] , init[4] = 0.04,0.25,0.25,0.5,0.25

        lowerBound = ql.Array(5)
        lowerBound[0] = 0.0
        lowerBound[1] = 0.01
        lowerBound[2] = 0.0
        lowerBound[3] = -1.0
        lowerBound[4] = 0.0

        upperBound = ql.Array(5)
        upperBound[0] = 1.0
        upperBound[1] = 15.0
        upperBound[2] = 1.0
        upperBound[3] = 1.0
        upperBound[4] = 1.0

        maxIterations = 10000
        minStatIterations = 9999 
        rootEpsilon = 1e-10
        functionEpsilon = 1e-10
        gradientNormEpsilon = 1e-10

        myEndCrit = ql.EndCriteria(maxIterations , minStatIterations , rootEpsilon , functionEpsilon ,
        gradientNormEpsilon)

        # constraint = ql.NoConstraint()
        constraint = ql.NonhomogeneousBoundaryConstraint(lowerBound , upperBound)
        
        er = lambda theta , kappa , sigma , rho , nu0 : self.heston_price_error_adjust(theta,sigma,rho,s,T,price_data,K,rd,rf,kappa,nu0)
        # method = ql.DifferentialEvolution()
        method = ql.Simplex(1.0)
        out = ql.Optimizer().solve(function=er,c=constraint,e=myEndCrit,m=method,iv=init)
        theta , kappa , sigma , rho , nu0 = np.array(out)
        if 2*theta * kappa >= (sigma*sigma) :
            '''
            Feller condition
            '''
            print('successfull...')
            print(self.heston_price_error_adjust(theta,sigma,rho,s,T,price_data,K,rd,rf,kappa,nu0))    
            return theta , kappa , sigma , rho , nu0
        else : 
            print('optimzation failed ..')
            return theta , kappa , sigma , rho , nu0

