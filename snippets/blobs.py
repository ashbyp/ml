from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

data, target = make_blobs(n_samples=500, n_features=3, centers=4,
                          shuffle=True, random_state=42, cluster_std=2.5)

print(data)
print(target)
print(data.shape)
print(target.shape)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=target, alpha=0.7, cmap="Set1")
plt.show()
