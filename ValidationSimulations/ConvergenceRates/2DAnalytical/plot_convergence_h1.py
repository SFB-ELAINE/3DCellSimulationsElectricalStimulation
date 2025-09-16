import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.rcParams["legend.framealpha"] = 1.0
show = True
min_order = 0
max_order = 3
filenames = [
    "results/results_order_{}".format(i + 1) for i in range(min_order, max_order)
]

data = {}

for f in filenames:
    tmp = pd.read_csv(f + ".csv")
    data[f] = tmp

plt.xscale("log")
plt.yscale("log")
plt.xlabel("# DOF")
plt.ylabel(r"H$_1$-error")
for f in filenames:
    order = f.split("_")[-1]
    ls = "s-"
    ndof = data[f]["ndofs"].to_numpy().astype(np.float64)
    nels = data[f]["nels"].to_numpy().astype(np.float64)
    h1 = data[f]["h1"].to_numpy().astype(np.float64)
    plt.plot(ndof, h1, ls, label="p={}".format(order))
    hestim = np.power(data[f]["ndofs"].to_numpy(), -0.5).astype(np.float64)
    slope, intercept, r, p, stderr = linregress(np.log10(hestim), np.log10(h1))
    print("Slope: ", slope)
    plt.plot(ndof, 10 ** (0.7 * intercept) * hestim**slope, "black")

plt.tight_layout()
plt.legend()
plt.savefig("h1comparedofs.pdf")
if show:
    plt.show()
else:
    plt.close()

plt.xscale("log")
plt.yscale("log")
plt.xlabel("# elements")
plt.ylabel(r"H$_1$-error")
for f in filenames:
    order = int(f.split("_")[-1])
    ls = "s-"
    ndof = data[f]["ndofs"].to_numpy().astype(np.float64)
    nels = data[f]["nels"].to_numpy().astype(np.float64)
    h1 = data[f]["h1"].to_numpy().astype(np.float64)
    plt.plot(nels, h1, ls, label="p={}".format(order))
    hestim = np.power(data[f]["nels"].to_numpy(), -0.5).astype(np.float64)
    slope, intercept, r, p, stderr = linregress(np.log10(hestim), np.log10(h1))
    print("Slope: ", slope)
    plt.plot(nels, 10 ** (0.7 * intercept) * hestim**slope, "black")

plt.tight_layout()
plt.legend()
plt.savefig("h1compareels.pdf")
if show:
    plt.show()
else:
    plt.close()
