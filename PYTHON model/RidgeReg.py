#%%
"""
Authors: Roser Cantenys Sabà, David Bergés Lladó

Python implementation using scipy.optimize of Ridge Regresssion problem

	min(w,gam) 		0.5 * (Aw +gam -y)' * (Aw +gam -y)
	s.to      		norm(w)^2 <= t

"""
#%%

import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


#%%

# File location:
mypath = Path().absolute()
# Set file location as current working directory:
os.chdir(mypath)

print('Current directory: ', os.getcwd())

#%%
# Load the data of our problem ('deathrate_instance_python.dat')

# File description:
with open('./../DATA/Deathrate/deathrate_instance_python.dat') as myfile:
    for i in range(43):
        print(next(myfile))

#%%

# Set up A (matrix of predictors) and y (response value)
A = np.loadtxt('./../DATA/Deathrate/deathrate_instance_python.dat')
y = np.array(A[:,15])
A = np.array(A[:,:15])

#%%

# Dimensions of the variables:
n = len(A); p = len(A[1])
print('y shape: ', np.shape(y))
print('A shpae: ', np.shape(A))

e = np.ones(n)

#%%
"""
OBJECTIVE FUNCTION INFORMATION (OBJECTIVE, GRADIENT, HESSIAN)

Parameters for all the functions: (x)
x := array of p+1 positions:
    - x[:p] := w
    - x[p]  := gamma
"""

# Objective function
# Returns the objective function evaluated at (x)
def ridge_obj(x):
    # Variables
    w     = x[:p]
    gamma = x[p]

    return(0.5*np.transpose(A@w +gamma -y)@(A@w +gamma -y))


# Gradient
# Returns the gradient of the objective function evaluated at (x)
def ridge_grad(x):
    # Variables
    w     = x[:p]
    gamma = x[p]

    At = np.transpose(A)
    et = np.transpose(e)

    grad     = np.zeros(p+1)
    grad[:p] = At@A@w +At@e*gamma -At@y #grad_w     (px1)
    grad[p]  = et@A@w +gamma*et@e -et@y #grad_gamma (1x1)

    return(grad)


# Hessian
# Returns the Hessian matrix of the objective function evaluated at (x)
def ridge_hess(x):
    # We don't need the parameters here
    At = np.transpose(A)
    et = np.transpose(e)

    H        = np.zeros((p+1,p+1))
    H[:p,:p] = At@A #grad_w2      (pxp)
    H[:p,p]  = At@e #grad_w_gamma (px1)
    H[p,:p]  = et@A #grad_gamma_w (1xp)
    H[p,p]   = et@e #grad_gamma2  (1x1)

    return(H)

#%%
"""
CONSTRAINT INFORMATION (norm(w)^2 <= t)

Parameters for all the functions: (x)
x := array of p+1 positions:
    - x[:p] := w
    - x[p]  := gamma

In the Hessian: v is array of size number of constraints
"""

# Constraint
# Returns the evaluation of the constraint
def ridge_cons_f(x):
    # We only need w
    w = x[:p]

    return(np.transpose(w)@w)


# Jacobian
# Returns the evaluation of the Jacobian of the constraint
def ridge_cons_J(x):
    # We only need w
    w = x[:p]

    grad     = np.zeros(p+1)
    grad[:p] = 2*w #grad_cons_w

    return(grad)


# Hessian
# The evaluation of: sum{i in 1..m} v[i]*H_i, where m is the number of
# constraints, and H_i the Hessian of the constraint i. In this case m=1, so
# we just return v[0]*H
def ridge_cons_H(x,v):
    # We don't need the parameters here
    H        = np.zeros((p+1,p+1))
    H[:p,:p] = np.diag(np.repeat(2,p)) #grad_cons_w2

    return(v[0]*H)

#%%

# CONSTRAINT

# We can tune the value of t here (upper bound of the constraint)
t = 1
# Non linear constraint
nonlinear_constraint = NonlinearConstraint(ridge_cons_f, lb = -np.inf, ub = t, jac = ridge_cons_J, hess = ridge_cons_H)

#%%

# PROBLEM

# We take a random initial point
x0 = np.random.rand(p+1)

sol = minimize(ridge_obj, x0 = x0, method = 'trust-constr', jac = ridge_grad, hess = ridge_hess, constraints = [nonlinear_constraint], bounds = None, options = {'verbose' : 1})
print(sol)

#%%


#%%
"""
EXAMPLE 2:

We can try our code with other datasets from Internet:

link: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

We are working with the 'winequality-red.csv' file

Brief variable description:
    Input variables (based on physicochemical tests):
        1 - fixed acidity
        2 - volatile acidity
        3 - citric acid
        4 - residual sugar
        5 - chlorides
        6 - free sulfur dioxide
        7 - total sulfur dioxide
        8 - density
        9 - pH
        10 - sulphates
        11 - alcohol
        Output variable (based on sensory data):
        12 - quality (score between 0 and 10)
"""
#%%

# Set up A (matrix of predictors) and y (response value)
A = np.loadtxt('./../DATA/Wines/wines_python.dat')
y = np.array(A[:,-1])
A = np.array(A[:,:-1])


#%%

# Dimensions of the variables:
n = len(A); p = len(A[1])
print('y shape: ', np.shape(y))
print('A shpae: ', np.shape(A))

e = np.ones(n)

#%%

# ADDITIONAL PROBLEM (we don't modify the contraint, as it is the same, we can tune the parameter t from above)

# We take a random initial point
x0 = np.random.rand(p+1)

sol = minimize(ridge_obj, x0 = x0, method = 'trust-constr', jac = ridge_grad, hess = ridge_hess, constraints = [nonlinear_constraint], bounds = None, options = {'verbose' : 1, 'maxiter' : 500})
print(sol)

def printsol(sol):
	print('Loss function value: ', sol.fun)
	print('Vector of coefficients: ', sol.x[:-1])
	print('Gamma: ', sol.x[-1])
	print('norm2_w: ', sol.v[0][0])

printsol(sol)
