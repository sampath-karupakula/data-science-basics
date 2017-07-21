# import the libraris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# load the data
df = pd.read_csv('data_1d.csv', sep=',', names=['X', 'Y'])

# convert the data into numpy arrays

X = df['X'].values
Y = df['Y'].values

plt.scatter(X, Y)
plt.show()

# y = mx + b
# calculate m and b
# m and b shares common denominator
dent = X.dot(X) - X.mean() * X.sum()
m = (X.dot(Y) - Y.mean() * X.sum()) / dent
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / dent

yHat = X * m + b

plt.scatter(X, Y)
plt.plot(X, yHat)
plt.show()

res = Y - yHat
tot = Y - Y.mean()

r2 = 1 - res.dot(res) / tot.dot(tot)
print(r2)
