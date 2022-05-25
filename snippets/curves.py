from sklearn.datasets import make_s_curve, make_swiss_roll
import matplotlib.pyplot as plt

data, target = make_s_curve(n_samples=5000, random_state=42)

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=target, alpha=0.7, cmap="viridis")

data, target = make_swiss_roll(n_samples=5000, random_state=42)
ax = fig.add_subplot(122, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=target, alpha=0.7, cmap="viridis")

plt.show()
