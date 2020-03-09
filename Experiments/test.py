import numpy as np

# true_s_values = np.load('true_values.npy')  # load from file
# binary = np.load('err_listbinary.npy')
# tile = np.load('err_listtilecoding.npy')
# agg = np.load('err_listsaggregation.npy')

x = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2])
mu = np.load('mu.npy')
print(mu.dot(x))


