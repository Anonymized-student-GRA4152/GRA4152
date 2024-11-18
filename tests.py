import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy.stats import norm
from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.optimize import minimize
import argparse
from GRA41522 import DataLoader, LoadSMdataset, Loadcsv
from GRA41522 import GLM, Normal, Bernoulli, Poisson


parser = argparse.ArgumentParser()
parser.add_argument("--add-intercept", action="store_true")
