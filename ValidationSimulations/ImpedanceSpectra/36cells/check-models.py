import impedancefitter as ifit
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0 as e0
import numpy as np
import glob

show = True

result_files = glob.glob("*Shell*/benchmark/*.csv")

# load simulation files into fitter library
fitter = ifit.Fitter("CSV", fileList=result_files)

plt.rcParams["legend.framealpha"] = 1.0
plt.rcParams["lines.linewidth"] = 3.0

# use outer dimensions of box to get unit capacitance
c0 = 100e-6 * e0

# initialize analytical models
ecm_models = {}
for result_file in result_files:
    if "SingleShell" in result_file:
        model_name = "SS"
        ecm_model = "SingleShell"
        if "Wall" in result_file:
            model_name = "SSW"
            ecm_model = "SingleShellWall"
    if "DoubleShell" in result_file:
        model_name = "DS"
        ecm_model = "DoubleShell"
        if "Wall" in result_file:
            model_name = "DSW"
            ecm_model = "DoubleShellWall"
    ecm_models[model_name] = ifit.get_equivalent_circuit_model(ecm_model)

# add parameters of analytical model

p = 0.05172318144  # volume ratio
Rc = 7e-6  # cell radius
Rn = 0.8 * Rc  # nucleus radius
dm = 7e-9  # membrane thickness
dn = 40e-9  # nuclear membrane thickness
dw = 2.5e-6  # cell wall thickness

sig_m = 8.7e-6  # membrane conductivity
sig_buf = 1.0  # ecm conductivity
sig_cyt = 0.48  # cytoplasm conductivity
sig_nm = 3e-3  # nuclear membrane conductivity
sig_np = 0.95  # nucleoplasm conductivity
sig_wall = 0.01  # cell wall conductivity

eps_m = 5.8  # membrane permittivity
eps_buf = 80  # ecm permittivity
eps_cyt = 60  # cytoplasm permittivity
eps_nm = 41  # nuclear membrane permittivity
eps_np = 120  # nucleoplasm permittivity
eps_wall = 20  # cell wall permittivity

# initialize parameter dict
# Note: some units have to be converted
# (e.g. from m to um)
parameters = {
    "em": eps_m,
    "km": sig_m / 1e-6,
    "kcp": sig_cyt,
    "ecp": eps_cyt,
    "kmed": sig_buf,
    "emed": eps_buf,
    "p": p,
    "c0": c0 / 1e-12,
    "dm": dm,
    "dn": dn,
    "dw": dw,
    "kne": sig_nm / 1e-3,
    "ene": eps_nm,
    "knp": sig_np,
    "enp": eps_np,
    "kw": sig_wall,
    "ew": eps_wall,
    "Rc": Rc,
    "Rn": Rn,
}

# update volume ratio for cell wall models
wall_parameters = parameters.copy()
wall_parameters["p"] = 0.129289

# frequencies
omega = 2.0 * np.pi * np.logspace(3, 12, num=91)

for result_file in result_files:
    if "SingleShell" in result_file:
        model_name = "SS"
        if "Wall" in result_file:
            model_name = "SSW"
    if "DoubleShell" in result_file:
        model_name = "DS"
        if "Wall" in result_file:
            model_name = "DSW"
    print(f"Plotting for model: {model_name}")
    print(result_file)
    numerical_Z = fitter.z_dict[result_file][0]
    if "W" in model_name:
        print("call wall model")
        analytical_Z = ecm_models[model_name].eval(omega=omega, **wall_parameters)
    else:
        analytical_Z = ecm_models[model_name].eval(omega=omega, **parameters)

    ifit.plot_impedance(
        omega,
        analytical_Z,
        Z_fit=numerical_Z,
        labels=["Analytical", "Numerical", ""],
        title=model_name,
        save=True,
        show=show,
    )
    plt.close()
    ifit.plot_dielectric_properties(
        omega,
        analytical_Z,
        c0,
        Z_comp=numerical_Z,
        labels=["Analytical", "Numerical"],
        title=model_name,
        save=True,
        show=show,
    )
    plt.close()
    ifit.plot_comparison_dielectric_properties(
        omega,
        analytical_Z,
        c0,
        Z_comp=numerical_Z,
        title=model_name,
        save=True,
        show=show,
        legend=False,
    )
    plt.close()
