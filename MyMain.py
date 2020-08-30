import numpy as np
np.set_printoptions(suppress=True)

from MyNetwork import Model

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_digits, load_iris
from sklearn.datasets.samples_generator import make_blobs, make_moons, make_regression,make_s_curve, make_friedman1
from matplotlib import pyplot
from pandas import DataFrame


def split_train_test_dataset(X, y):
    # size of X : [num_input x num_dimension]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    train_X = train_X.T
    test_X = test_X.T

    return train_X, train_y, test_X, test_y


def normalize_data(train_X, test_X):
    means = np.mean(train_X, axis=1, keepdims=True)
    std_dev = np.std(train_X, axis=1, keepdims=True)

    train_X = (train_X - means) / std_dev
    test_X = (test_X - means) / std_dev

    return train_X, test_X, means, std_dev


# generate  classification dataset
#X_1,y_1 = make_friedman1(n_samples=1000)
#X_s_1,y_1 = make_s_curve(n_samples=1000)
#X_1,y_1 = make_regression(n_samples=1000)
#X_1,y_1 = make_moons(n_samples=1000)
X_1, y_1 = make_blobs(n_samples=5000, centers=2, n_features=2)
#print(X_1)
y_1 = np.array(y_1).reshape((len(y_1),1))

#print(y_1.squeeze())
#print(X_1[:,0])

# # scatter plot, dots colored by class value
# df = DataFrame(dict(x=X_1[:,0], y=X_1[:,1], label=y_1.squeeze()))
# colors = {0:'red', 1:'blue', 2:'green'}
# fig, ax = pyplot.subplots()
# grouped = df.groupby('label')
# print(X_1.shape, y_1.shape)
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
# pyplot.show()

train_X, train_y, test_X, test_y = split_train_test_dataset(X_1,y_1)
before_inference = test_X

train_y = train_y.reshape((1, len(train_y)))
test_y = test_y.reshape((1, len(test_y)))

# normalize_data
train_X, test_X, means, std_dev = normalize_data(train_X, test_X)

Model_description = [{"layer_size" : 128, "activation" : "sigmoid"},
               {"layer_size" : 128, "activation" : "sigmoid"},
               {"layer_size" : 1, "activation" : "sigmoid"}]

model = Model(Model_description, 2, "cross_entropy_sigmoid", train_X, train_y, learning_rate=0.01)

history = model.train(100)
print("acc:", model.calculate_accuracy(test_X, test_y))


pred = model.run_inference(test_X)
#print(before_inference)
#print(before_inference[0,:])
#print("pred:",pred[0])
#print(before_inference[1,:])



# scatter plot, dots colored by class value

df = DataFrame(dict(x=before_inference[0,:], y=before_inference[1,:], label=pred[0]))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

df = DataFrame(dict(x=before_inference[0,:], y=before_inference[1,:], label=test_y.squeeze()))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

plt.plot(history)
pyplot.show()
