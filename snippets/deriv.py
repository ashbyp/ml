#https://machinelearningmastery.com/gradient-in-machine-learning/

# plot of simple function
from numpy import arange
from matplotlib import pyplot


# objective function
def objective(x):
	return x**2.0


# define range for input
r_min, r_max = -1.0, 1.0
# sample input range uniformly at 0.1 increments
inputs = arange(r_min, r_max+0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# show the plot
pyplot.show()

# calculate the derivative of the objective function

# derivative of objective function
def derivative(x):
	return x * 2.0

# calculate derivatives
d1 = derivative(-0.5)
print('f\'(-0.5) = %.3f' % d1)
d2 = derivative(0.5)
print('f\'(0.5) = %.3f' % d2)
d3 = derivative(0.0)
print('f\'(0.0) = %.3f' % d3)