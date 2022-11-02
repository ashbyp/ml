import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()

data, target = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)

print(data)
print(target)

data["target"] = target

print(type(data))
print(type(target))

sns.pairplot(data, kind="scatter", diag_kind="kde", hue="target",
             palette="muted", plot_kws={'alpha': 0.7})

plt.show()
