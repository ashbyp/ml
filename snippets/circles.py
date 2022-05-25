from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

data, target = make_circles(n_samples=500, shuffle=True, factor=0.7, noise=0.1)
plt.figure(figsize=(6,6))
plt.scatter(data[:,0], data[:,1], c=target, alpha=0.8, cmap="Set1")
plt.show()

print(target)