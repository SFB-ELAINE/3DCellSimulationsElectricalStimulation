import os
import ngsolve as ng

# needed for analytical solution
from netgen.geom2d import SplineGeometry
from sympysolution import return_solution
from sympy import diff
import pandas as pd
from cell_simulation_utilities import ThinLayerFunctionSpace


ng.SetNumThreads(12)
# enable logging level
# ng.ngsglobals.msg_level = 10

# maximum element size
maxh = 0.5

# cofficients for the subdomains
ki = 1.0
ke = 100.0
# Assigns material properties
sigvals = {"in": ki, "out": ke}

# interface resistance
R = 0.5

# get analytical solution on interior and exterior subdomain as well as jump
soli, sole, jumpana = return_solution()
cf_i = ng.CoefficientFunction(eval(str(soli)))
bnd_cf = {"outer": eval(str(sole))}
cf_e = ng.CoefficientFunction(eval(str(sole)))
jump_ana = ng.CoefficientFunction(eval(str(jumpana)))
solution = ng.CoefficientFunction([cf_e, cf_i])

# gradient of analytical solution
gradcf_e = ng.CoefficientFunction(
    (eval(str(diff(sole, "x"))), eval(str(diff(sole, "y"))))
)
gradcf_i = ng.CoefficientFunction(
    (eval(str(diff(soli, "x"))), eval(str(diff(soli, "y"))))
)

gradient_solution = ng.CoefficientFunction([gradcf_e, gradcf_i])

geo = SplineGeometry()
geo.AddCircle(c=(0, 0), r=1.5, bc="outer", leftdomain=1, rightdomain=0)
geo.AddCircle(c=(0, 0), r=1.0, bc="interface", leftdomain=2, rightdomain=1)
geo.SetMaterial(1, "out")
geo.SetMaterial(2, "in")

# discretization order (used for FESpace and for geometry approximation):
for order in [1, 2, 3]:
    for curved in [True, False]:
        ngmesh = geo.GenerateMesh(maxh=maxh, quad_dominated=False)
        mesh = ng.Mesh(ngmesh)
        # initial refine
        mesh.Refine()
        if curved:
            mesh.Curve(order)

        l2errors = []
        h1errors = []
        h1semierrors = []
        ndofs = []
        nels = []
        jumperr = []
        conditions = []

        n_ref = 5
        for i in range(n_ref):
            print(f"Iteration {i}")
            if i > 0:
                print("Refine mesh")
                mesh.Refine()
                if curved:
                    print("Curve mesh")
                    mesh.Curve(order)

            fes = ThinLayerFunctionSpace(
                mesh,
                domain_outside="out",
                domain_inside="in",
                interface="interface",
                dirichlet="outer",
                order=order,
            )

            sig = mesh.MaterialCF(sigvals, default=(0, 0, 0))

            ndofs.append(fes.fes.ndof)
            nels.append(fes.mesh.ne)

            print("Get Bilinear form")
            a, f = fes.get_bilinear_form(sig, 1.0 / R)

            # set boundary
            fes.set_boundary_condition(bnd_cf)
            with ng.TaskManager():
                # apply BCs
                print("Solving")

                fes.direct_solver(a, f)

                print("Evaluating solution")
                errors = fes.compute_error(solution, gradient_solution)
                print("TMP error")
                tmperr = fes.compute_jump_error(jump_ana)
                print("Done")

            l2errors.append(errors["L2"])
            h1errors.append(errors["Energy"])
            h1semierrors.append(errors["H1"])
            jumperr.append(tmperr)

        # write results
        if not os.path.exists("results"):
            os.makedirs("results")
        results = {
            "nels": nels,
            "ndofs": ndofs,
            "h1": h1errors,
            "h1semi": h1semierrors,
            "l2": l2errors,
            "jump": jumperr,
        }
        df = pd.DataFrame(results)
        filename = "results/results_order"
        if not curved:
            filename += "_uncurved"
        filename += "_{}.csv".format(order)
        df.to_csv(filename, index=False)

        print("Convergence rates energy errors:")
        for i in range(1, len(h1errors)):
            print(ng.log(h1errors[i] / h1errors[i - 1]) / ng.log(0.5))
        print("Convergence rates h1-seminorm errors:")
        for i in range(1, len(h1semierrors)):
            print(ng.log(h1semierrors[i] / h1semierrors[i - 1]) / ng.log(0.5))
        print("Convergence rates l2 errors:")
        for i in range(1, len(l2errors)):
            print(ng.log(l2errors[i] / l2errors[i - 1]) / ng.log(0.5))
        print("Convergence rates jump errors:")
        for i in range(1, len(jumperr)):
            print(ng.log(jumperr[i] / jumperr[i - 1]) / ng.log(0.5))

        print("Success")
