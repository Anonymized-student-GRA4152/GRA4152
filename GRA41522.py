#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy.stats import norm
from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.optimize import minimize


# 2 Implementing a Hierarchy of Data Loaders for Different File Formats

# In[3]:


class DataLoader:
    def __init__(self, LoaderType):
        self._LoaderType = LoaderType

    def load_dataset(self):
        raise NotImplementedError
    
    def add_constant(self):
        return np.hstack([x, np.ones([x.shape[0], 1], dtype=np.int32)])

    def set_x(self, x):
        self._x = x
        return self._x
    
    def set_y(self, y):
        self._y = y
        return self._y
    
    @property
    def x_transpose(self):
        return np.transpose(self._x)

class LoadSMdataset(DataLoader):
    def __init__(self, LoaderType, dataset_name):
        super().__init__(LoaderType)
        self.dataset_name = dataset_name

    def load_dataset(self):
        data = sm.datasets.get_rdataset(self.dataset_name).data
        print('Here is the {} dataset from statsmodel.'.format(self.dataset_name))
        return data
        
class Loadcsv(DataLoader):
    def __init__(self, LoaderType, file_path): 
        super().__init__(LoaderType) 
        self.file_path = file_path
        
    def load_dataset(self):
        data = pd.read_csv(self.file_path)
        print('Here is the dataset')
        return data 


# 1. Implementing a Hierarchy of GLMs

# In[41]:


class GLM:
    #Construct a general GLM class with no specific type of distribution
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._betas = np.repeat(0.1, x.shape[1])
    
    @property
    def beta(self):
        return self._betas
    #Beta should be getter property since we don't want anyone to have access and change it directly, 
    # which will mess with our program

class Normal(GLM):
    def __init__(self, x, y):
        super().__init__(x,y)

    def negllik_normal(self, params, x, y):
        eta = np.matmul(x, params)
        #Using link function as in Table 1. For Normal Distribution, eta = Identity matric of mu
        mu = eta
        llik = np.sum(norm.logpdf(y,mu))
        return llik*(-1)
    #Next, we make a fit() method which is basically a maximum likelihood function, 
    # by minimizing the negative log-likelihood function
    def fit(self):
        results = minimize(self.negllik_normal , self._betas, args =(self._x, self._y))
        self._betas = results["x"]
    #Here we override beta() method in the superclass with our specific beta achived from our fit() method.
    #When user want to see beta, this is the beta that will be shown for Normal Distribution test on the
    #dataset, due to dynamic look up.
    def predict(self):
        #Using self._betas, which is calculated from the previous method, and matric x variables 
        self.predictmu = np.matmul(x, self._betas)
        #It is simply similar to how we find mu 
        return self.predictmu
    

        
class Bernoulli(GLM):
    def __init__(self,x,y):
        super().__init__(x,y)
        
    def negllik_ber(self,params, x, y):
        eta = np.matmul(x, params)
        #Using link function from Table 1. 
        #Equation: eta = ln(mu/(1-mu)). After some math this is the derived func to find mu:
        mu = (np.exp(eta))/(1+np.exp(eta))
        llik = np.sum(bernoulli.logpmf(y, mu))
        return llik*(-1)
    #We also we make a fit() method which is basically a maximum likelihood function, 
    #by minimizing the negative log-likelihood function
    def fit(self):
        results = minimize(self.negllik_ber, self._betas, args =(self._x, self._y))
        self._betas = results["x"]
    #Here we override beta method in the superclass with our specific beta achived from our fit() method.
    #When user want to gwt beta, this is the value that will be shown for beta for Normal Distribution test on the
    #dataset, due to dynamic look up.
    def predict(self, x):
        eta = np.matmul(x, self._betas)
        predicted_mu = (np.exp(eta))/(1+np.exp(eta))
        #It is simply similar to how we find mu. Unlike in Normal Distribution, where eta = Identity matric of mu,
        #here for Bernoulli Distribution, we have to find mu by formula in Table 1, and to do that,
        #we need to find the estimated eta again, this time we pass on value of beta that we
        #found through inimization of the negative maximum likelihood,
        #after that we do some calculation to find predicted mu
        return predicted_mu
    
    
class Poisson(GLM):
    def __init__(self,x,y):
        super().__init__(x,y)
    
    def negllik_poisson(self, params, x, y):
        eta = np.matmul(x, params)
        #Using link function from Table 1. 
        #Equation: eta = ln(mu). After some math here is the derived func to find mu:
        mu = np.exp(eta)
        llik = np.sum(poisson.logpmf(y, mu))
        return llik*(-1)
    #Again, we make a fit() method which is basically a maximum likelihood function, 
    #by minimizing the negative log-likelihood function
    def fit(self):
        results = minimize(self.negllik_poisson, self._betas, args =(self._x, self._y))
        self._betas = results["x"]
    #Here we override beta method in the superclass with our specific beta achived from our fit() method.
    #When user want to see beta, this is the beta that will be shown for Poisson Distribution test on the dataset.
    def predict(self, x):
        eta = np.matmul(x, self._betas)
        predicted_mu = np.exp(eta)
        #It is simply similar to how we find mu. We find eta again, this time with the minimized beta found 
        #through MLE. Then we calculate predicted mu. 
        return predicted_mu

