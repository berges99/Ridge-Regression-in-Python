# Parameters
param n >= 1, integer;   # features
param m >= 1, integer;   # observations
param t >= 0;            # constrain upper bound

param y {1..m};          # response value
param A {1..m,1..n};     # data matrix

# Variables
var gamma;
var w {1..n};

# Constrained Ridge Regression model (regression with Tikhonov or L2 regularization)
minimize ridge_reg:
    0.5*sum{i in 1..m}((sum{j in 1..n} A[i,j]*w[j]) +gamma -y[i])^2;
subject to norm2_w:
    sum{j in 1..n} w[j]^2 <= t;