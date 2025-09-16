import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.rcParams["legend.framealpha"] = 1.0
show = True
curvecontinue = False
min_order = 0
max_order = 3
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
plt.ylabel(r"L$_2$-error")
for f in filenames:
    order = f.split("_")[-1]
    curve = ", curved"
    ls = "o-"
    if "uncurved" in f:
        if curvecontinue:
            continue
        curve = ""
        ls = "o"
    ndof = data[f]["ndofs"].to_numpy().astype(np.float64)
    nels = data[f]["nels"].to_numpy().astype(np.float64)
    l2 = data[f]["l2"].to_numpy().astype(np.float64)
    plt.plot(ndof, l2, ls, label="p={}".format(order) + curve)
    if curve != "":
        hestim = np.power(data[f]["ndofs"].to_numpy(), -0.5).astype(np.float64)
        slope, intercept, r, p, stderr = linregress(np.log10(hestim), np.log10(l2))
        print("Slope: ", slope)
        plt.plot(ndof, 10 ** (0.7 * intercept) * hestim**slope, "black")

plt.tight_layout()
plt.legend()
plt.savefig("l2comparedofs.pdf")
if show:
    plt.show()
else:
    plt.close()

plt.xscale("log")
plt.yscale("log")
plt.xlabel("# elements")
plt.ylabel(r"L$_2$-error")
for f in filenames:
    order = int(f.split("_")[-1])
    curve = ", curved"
    ls = "s-"
    if "uncurved" in f:
        if curvecontinue:
            continue
        curve = ""
        ls = "o"
    ndof = data[f]["ndofs"].to_numpy().astype(np.float64)
    nels = data[f]["nels"].to_numpy().astype(np.float64)
    l2 = data[f]["l2"].to_numpy().astype(np.float64)
    plt.plot(nels, l2, ls, label="p={}".format(order) + curve)
    if curve != "":
        hestim = np.power(data[f]["nels"].to_numpy(), -0.5).astype(np.float64)
        slope, intercept, r, p, stderr = linregress(np.log10(hestim), np.log10(l2))
        print("Slope: ", slope)
        plt.plot(nels, 10 ** (0.7 * intercept) * hestim**slope, "black")

plt.tight_layout()
plt.legend()
plt.savefig("l2compareels.pdf")
if show:
    plt.show()
else:
    plt.close()
