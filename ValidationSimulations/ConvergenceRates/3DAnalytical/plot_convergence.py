import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

show = False
show = True
curvecontinue = False
curvecontinue = True
min_order = 0
max_order = 2
filenames = [
    "results/results_order_{}".format(i + 1) for i in range(min_order, max_order)
]
filenames += [
    "results/results_order_uncurved_{}".format(i + 1)
    for i in range(min_order, max_order)
]

data = {}

for f in filenames:
    tmp = pd.read_csv(f + ".csv")
    data[f] = tmp

plt.xscale("log")
plt.yscale("log")
plt.xlabel("# DOF")
plt.ylabel(r"TMP-error")
for f in filenames:
    order = int(f.split("_")[-1])
    curve = ", curved"
    ls = "o-"
    if "uncurved" in f:
        if curvecontinue:
            continue
        curve = ""
        ls = "o"
    ndof = data[f]["ndofs"].to_numpy().astype(np.float64)
    nels = data[f]["nels"].to_numpy().astype(np.float64)
    h1 = data[f]["tmp"].to_numpy().astype(np.float64)
    plt.plot(ndof, h1, ls, label="p={}".format(order) + curve)
    if curve != "":
        hestim = np.power(data[f]["ndofs"].to_numpy(), -0.3333).astype(np.float64)
        slope, intercept, r, p, stderr = linregress(np.log10(hestim), np.log10(h1))
        print("Slope: ", slope)
        plt.plot(ndof, 10 ** (0.7 * intercept) * hestim ** (order + 1), "black")

plt.tight_layout()
plt.legend()
plt.savefig("tmpcomparedofs.pdf")
if show:
    plt.show()
else:
    plt.close()

plt.xscale("log")
plt.yscale("log")
plt.xlabel("# elements")
plt.ylabel(r"TMP-error")
for f in filenames:
    order = int(f.split("_")[-1])
    curve = ", curved"
    ls = "o-"
    if "uncurved" in f:
        if curvecontinue:
            continue
        curve = ""
        ls = "o"
    ndof = data[f]["ndofs"].to_numpy().astype(np.float64)
    nels = data[f]["nels"].to_numpy().astype(np.float64)
    h1 = data[f]["tmp"].to_numpy().astype(np.float64)
    plt.plot(nels, h1, ls, label="p={}".format(order) + curve)
    if curve != "":
        hestim = np.power(data[f]["nels"].to_numpy(), -0.3333).astype(np.float64)
        slope, intercept, r, p, stderr = linregress(np.log10(hestim), np.log10(h1))
        print("Slope: ", slope)
        print(intercept)
        if intercept < 0:
            intercept = 1
        plt.plot(nels, 10**intercept * hestim ** (order + 1), "black")

plt.tight_layout()
plt.legend()
plt.savefig("tmpcompareels.pdf")
if show:
    plt.show()
else:
    plt.close()
