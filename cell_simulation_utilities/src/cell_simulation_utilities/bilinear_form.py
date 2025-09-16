import ngsolve as ng
from typing import List, Optional

try:
    from ngsPETSc import KrylovSolver, Matrix, PETScPreconditioner

    HasPETSc = True
except ImportError:
    HasPETSc = False
import random
import networkx as nx


class ThinLayerFunctionSpace:
    def __init__(
        self,
        mesh: ng.Mesh,
        domain_outside: str,
        domain_inside: str,
        interface: str,
        dirichlet: str,
        order: int = 1,
        complex: bool = False,
        plateaus: Optional[list[str]] = None,
    ):
        self.mesh = mesh
        if self.mesh.GetCurveOrder() > order:
            raise ValueError(
                "The mesh is curved"
                " with a higher order"
                " than the function space"
                " polynomial order."
            )
        self.domain_outside = domain_outside
        self.domain_inside = domain_inside
        self.interface = interface
        self.complex = complex
        self._set_function_space(order, dirichlet, plateaus)
        self._set_grid_function()
        self._initial_elements = mesh.ne
        self._petsc_solver = None
        self._preconditioner = None

    def _set_function_space(
        self, order: int, dirichlet: str, plateaus: Optional[List[str]] = None
    ) -> ng.FESpace:
        """Define a function space considering dirichlet BCs."""
        fes_out = ng.H1(
            self.mesh,
            order=order,
            definedon=self.domain_outside,
            definedonbound=self.interface,
            dirichlet=dirichlet,
            complex=self.complex,
        )
        if plateaus is not None:
            ngs_plateaus = [self.mesh.Boundaries(plateau) for plateau in plateaus]
            fes_out = ng.PlateauFESpace(fes_out, ngs_plateaus)
        fes_in = ng.H1(
            self.mesh,
            order=order,
            definedon=self.domain_inside,
            definedonbound=self.interface,
            complex=self.complex,
        )
        fes = ng.FESpace([fes_out, fes_in])
        self.fes = ng.CompressCompound(fes)

    def get_bilinear_form(self, sigma: ng.CF, interface_admittance: ng.CF):
        """Formulate bilinear form for given conductivity and interface admittance."""

        (u_out, u_in) = self.fes.TrialFunction()
        (v_out, v_in) = self.fes.TestFunction()

        a = ng.BilinearForm(self.fes)

        a += sigma * ng.grad(u_out) * ng.grad(v_out) * ng.dx(self.domain_outside)
        a += sigma * ng.grad(u_in) * ng.grad(v_in) * ng.dx(self.domain_inside)

        # Interface condition
        a += (
            interface_admittance
            * (u_out - u_in)
            * (v_out - v_in)
            * ng.ds(self.interface)
        )

        # is zero, no further definition needed
        f = ng.LinearForm(self.fes)

        return a, f

    def _set_grid_function(self):
        """Get grid function on function space."""
        self.gfu = ng.GridFunction(self.fes)

    def compute_error(self, analytical_solution, grad_analytical_solution):
        """Compute the error in L2 and energy (H1) norm."""
        gfu_out, gfu_in = self.gfu.components
        mesh = self.gfu.space.mesh
        order = self.gfu.space.globalorder
        u_tmp = {self.domain_outside: gfu_out, self.domain_inside: gfu_in}
        u = mesh.MaterialCF(u_tmp)
        gradu_tmp = {
            self.domain_outside: ng.grad(gfu_out),
            self.domain_inside: ng.grad(gfu_in),
        }
        gradu = mesh.MaterialCF(gradu_tmp)

        error = u - analytical_solution
        l2errout = ng.Integrate(
            error * ng.Conj(error),
            mesh,
            definedon=mesh.Materials(self.domain_outside),
            order=2 * order,
        )
        l2errin = ng.Integrate(
            error * ng.Conj(error),
            mesh,
            definedon=mesh.Materials(self.domain_inside),
            order=2 * order,
        )

        graderror = gradu - grad_analytical_solution
        # energy norm
        h1seminormout = ng.Integrate(
            graderror * ng.Conj(graderror),
            mesh,
            definedon=mesh.Materials(self.domain_outside),
            order=2 * order,
        )
        energyerrout = l2errout + h1seminormout

        h1seminormin = ng.Integrate(
            graderror * ng.Conj(graderror),
            mesh,
            definedon=mesh.Materials(self.domain_inside),
            order=2 * order,
        )

        energyerrin = l2errin + h1seminormin

        errors = {
            "L2": ng.sqrt(l2errout + l2errin).real,
            "L2_in": l2errin.real,
            "L2_out": l2errout.real,
            "Energy": ng.sqrt(energyerrout + energyerrin).real,
            "Energy_in": energyerrin.real,
            "Energy_out": energyerrout.real,
            "H1": ng.sqrt(h1seminormout + h1seminormin).real,
            "H1_in": h1seminormin.real,
            "H1_out": h1seminormout.real,
        }
        return errors

    def compute_jump_error(self, jump_analytical: ng.CF):
        """Compute the error of the interface jump."""
        gfu_out, gfu_in = self.gfu.components
        mesh = self.gfu.space.mesh
        order = self.gfu.space.globalorder
        # jump error
        fes1 = ng.FacetFESpace(
            mesh,
            order=order,
            definedon=mesh.Materials(self.domain_outside),
            complex=self.complex,
        )
        gf1 = ng.GridFunction(fes1)
        gf1.Set(
            ng.BoundaryFromVolumeCF(gfu_out), definedon=mesh.Boundaries(self.interface)
        )

        fes2 = ng.FacetFESpace(
            mesh,
            order=order,
            definedon=mesh.Materials(self.domain_inside),
            complex=self.complex,
        )
        gf2 = ng.GridFunction(fes2)
        gf2.Set(
            ng.BoundaryFromVolumeCF(gfu_in), definedon=mesh.Boundaries(self.interface)
        )
        jump = gf2 - gf1
        error = jump - jump_analytical
        tmperr = ng.Integrate(
            error * ng.Conj(error),
            mesh,
            definedon=mesh.Boundaries(self.interface),
            order=2 * order,
        )
        return ng.sqrt(tmperr).real

    def set_boundary_condition(self, bnd_dict: dict) -> None:
        """Go through components of GridFunction and set boundary value."""
        if not set(bnd_dict.keys()).issubset(self.mesh.GetBoundaries()):
            raise ValueError(
                "Provided boundary condition for boundary not in mesh."
                f" Available BCs: {self.mesh.GetBoundaries()}"
            )
        bnd_cf = self.mesh.BoundaryCF(bnd_dict)
        print(bnd_dict)
        if getattr(
            self.gfu, "components", None
        ):  # checks if components exists and is non-empty
            for component in self.gfu.components:
                print("Setting component")
                component.Set(bnd_cf, ng.BND)
        else:
            print("Setting gfu directly")
            self.gfu.Set(bnd_cf, ng.BND)

    def direct_solver(self, a: ng.BilinearForm, f: ng.LinearForm):
        a.Assemble()
        f.Assemble()
        rhs = self.gfu.vec.CreateVector()
        rhs.data = f.vec - a.mat * self.gfu.vec
        update = self.gfu.vec.CreateVector()
        update.data = a.mat.Inverse(self.fes.FreeDofs(), inverse="sparsecholesky") * rhs
        self.gfu.vec.data += update

    def iterative_solver(
        self,
        a: ng.BilinearForm,
        f: ng.LinearForm,
        solver_type: str,
        preconditioner: str = None,
        solver_kwargs: dict = {},
        preconditioner_kwargs: dict = {},
    ):
        if preconditioner == "PETScPC" and not HasPETSc:
            raise RuntimeError("Please install ngsPETSc")

        if self._preconditioner is None:
            # SetUp preconditioner externally to avoid memory issues
            if preconditioner == "PETScPC":
                # ugly but we need to pass an assembled matrix
                a.Assemble()
                self._preconditioner = PETScPreconditioner(
                    a.mat, self.fes.FreeDofs(), preconditioner_kwargs
                )
            else:
                self._preconditioner = ng.Preconditioner(
                    a, preconditioner, **preconditioner_kwargs
                )
                a.Assemble()
            # Assemble once, will never change
            f.Assemble()
        else:
            if preconditioner == "PETScPC":
                # update existing preconditioner to avoid memory issue
                a.Assemble()
                if hasattr(a.mat, "row_pardofs"):
                    dofs = a.mat.row_pardofs
                else:
                    dofs = None
                matType = "aij"
                petscMat = Matrix(a.mat, (dofs, self.fes.FreeDofs(), None), matType).mat
                self._preconditioner.petscPreconditioner.setOperators(petscMat)
                self._preconditioner.petscPreconditioner.setUp()
                self._preconditioner.petscVecX, self._preconditioner.petscVecY = (
                    petscMat.createVecs()
                )
            else:
                a.Assemble()
        if solver_type.lower() == "cg":
            ng.krylovspace.CG(
                mat=a.mat,
                pre=self._preconditioner,
                rhs=f.vec,
                sol=self.gfu.vec,
                initialize=False,
                **solver_kwargs,
            )
        elif solver_type.lower() == "gmres":
            raise NotImplementedError("We haven't yet tested the GMRes solver.")
        else:
            raise ValueError("Available solvers: CG or GMRes")

    def petsc_solver(self, a: ng.BilinearForm, f: ng.LinearForm, petsc_opts: dict):
        if not HasPETSc:
            raise RuntimeError("Please install ngsPETSc")
        if self._petsc_solver is None:
            self._petsc_solver = KrylovSolver(a, self.fes, solverParameters=petsc_opts)
        else:
            a.Assemble()
            f.Assemble()
            A = Matrix(a.mat, self.fes).mat
            self._petsc_solver.ksp.setOperators(A=A)
        rhs = self.gfu.vec.CreateVector()
        # homogenization of boundary data and solution of linear system
        rhs.data = f.vec - a.mat * self.gfu.vec
        # TODO non-zero guess respected?
        gfu_temp = ng.GridFunction(self.fes)
        self._petsc_solver.solve(rhs, gfu_temp.vec)
        self.gfu.vec.data += gfu_temp.vec.data


class ShelledThinLayerFunctionSpace(ThinLayerFunctionSpace):
    def _iter_true_interfaces(self):
        """
        Yields (boundary_name, A, B) for boundaries whose value is a 2-tuple (A,B).
        Skips any singleton entries like 'Cell_bottom_*': 'Cell_i'.
        """
        for bnd, val in self.outer_inner_pairs.items():
            if isinstance(val, tuple) and len(val) == 2:
                A, B = val
                yield bnd, A, B

    def _iter_single_sided(self):
        """
        Yields (boundary_name, A) for boundaries whose value is a single domain string.
        E.g. electrode/bottom facets where you want TMP=0 (or some known trace).
        """
        for bnd, val in self.outer_inner_pairs.items():
            if isinstance(val, str):
                yield bnd, val

    def _union_regex(self, names):
        return "|".join(names) if names else "____none____"

    def _print_color_report(self):
        print("Color classes:")
        for k, cls in enumerate(getattr(self, "color_classes", [])):
            print(f"  C{k}: {cls}")

    def _build_nx_graph(self):
        if nx is None:
            raise ImportError("networkx is not installed. `pip install networkx`")
        G = nx.Graph()
        G.add_nodes_from(self.domains)
        for bnd, A, B in self._iter_true_interfaces():  # <-- only true interfaces
            if A in self.domains and B in self.domains and A != B:
                G.add_edge(A, B)
        return G

    def _apply_coloring_result(self, color_of: dict):
        """Pack color dict -> classes + mat2color (0..K-1), with validation."""
        if not color_of:
            self.color_classes = []
            self.mat2color = {}
            return
        # reindex to consecutive 0..K-1
        palette = {}
        nextc = 0
        compact = {}
        for m, c in color_of.items():
            if c not in palette:
                palette[c] = nextc
                nextc += 1
            compact[m] = palette[c]
        K = nextc
        classes = [[] for _ in range(K)]
        for m, c in compact.items():
            classes[c].append(m)

        # validate: neighbors differ
        conflicts = []
        for bnd, A, B in self._iter_true_interfaces():  # <-- change
            if A in compact and B in compact and compact[A] == compact[B]:
                conflicts.append((bnd, A, B, compact[A]))
        if conflicts:
            msg = "\n".join(
                f"  {n}: {A}-{B} both color {c}" for n, A, B, c in conflicts
            )
            raise ValueError("Invalid coloring (same-color neighbors):\n" + msg)

        self.color_classes = classes
        self.mat2color = compact

    def _nx_color_best(self, attempts: int = 10, seed: int | None = 0):
        """
        Try several NetworkX greedy strategies + a few randomized orders,
        keep the one with the fewest colors.
        """
        G = self._build_nx_graph()

        strategies = [
            "saturation_largest_first",  # DSATUR – usually best
            "largest_first",
            "smallest_last",
            "independent_set",
            "connected_sequential_bfs",
            "connected_sequential_dfs",
        ]

        def apply(strategy_name: str):
            return nx.coloring.greedy_color(G, strategy=strategy_name)

        def strategy_from_order(order):
            # greedy_color accepts a callable that returns the node ordering
            return lambda _G, _colors: order

        best = None
        best_k = 10**9
        best_name = None

        # deterministic strategies
        for strat in strategies:
            color = apply(strat)
            K = 1 + max(color.values()) if color else 0
            if K < best_k:
                best, best_k, best_name = color, K, strat

        # randomized orders
        if attempts > 0:
            rng = random.Random(seed)
            nodes = list(G.nodes())
            for t in range(attempts):
                order = nodes[:]
                rng.shuffle(order)
                color = nx.coloring.greedy_color(G, strategy=strategy_from_order(order))
                K = 1 + max(color.values()) if color else 0
                if K < best_k:
                    best, best_k, best_name = color, K, f"random_order[{t}]"

        self._apply_coloring_result(best)
        print(
            f"NetworkX coloring picked: {best_name} → {len(self.color_classes)} colors"
        )
        for k, cls in enumerate(self.color_classes):
            print(f"  C{k}: {cls}")

    def __init__(
        self,
        mesh: ng.Mesh,
        domains: List[str],
        interface_info: dict,
        outer_inner_pairs: dict,
        dirichlet: dict,
        order: int = 1,
        complex: bool = False,
        plateaus: dict[List[str]] = {},
    ):
        self.mesh = mesh
        if self.mesh.GetCurveOrder() > order:
            raise ValueError(
                "The mesh is curved"
                " with a higher order"
                " than the function space"
                " polynomial order."
            )
        self.domains = sorted(domains)
        self.enum_domains = {}
        for i, domain in enumerate(self.domains):
            self.enum_domains[domain] = i

        self.interface_info = interface_info
        self.outer_inner_pairs = outer_inner_pairs

        self._nx_color_best(attempts=10, seed=0)

        self._print_color_report()
        self.complex = complex
        self._set_function_space(order, dirichlet, plateaus)
        self._set_grid_function()
        self._initial_elements = mesh.ne
        self._petsc_solver = None
        self._preconditioner = None
        # GFUs used for TMP estimate
        self._gf1 = None
        self._gf2 = None

    @property
    def interfaces(self) -> list:
        return list(self.outer_inner_pairs.keys())

    def _set_function_space(
        self, order: int, dirichlet: dict, plateaus: dict[List[str]] = {}
    ):
        """Build a *few* spaces (one per color class) instead of one per domain."""
        fes_list = []

        for cls in self.color_classes:
            if not cls:
                continue

            # union of materials in this color
            definedon = self.mesh.Materials(self._union_regex(cls))

            # merge Dirichlet boundary regex for domains in this color (if provided)
            dir_parts = [dirichlet[d] for d in cls if d in dirichlet and dirichlet[d]]
            dir_str = self._union_regex(dir_parts) if dir_parts else None

            base_params = {
                "order": order,
                "definedon": definedon,
                "complex": self.complex,
            }
            if dir_str:
                base_params["dirichlet"] = dir_str

            base_h1 = ng.H1(self.mesh, **base_params)

            # merge Plateau lists (optional)
            merged_plateaus = []
            for d in cls:
                if d in plateaus and plateaus[d]:
                    merged_plateaus.extend(plateaus[d])

            if merged_plateaus:
                ngs_plateaus = [self.mesh.Boundaries(b) for b in merged_plateaus]
                fes_list.append(ng.PlateauFESpace(base_h1, ngs_plateaus))
            else:
                fes_list.append(base_h1)

        fes = ng.FESpace(fes_list)
        self.fes = ng.CompressCompound(fes)

        # facet space (scope it if you want only specific interfaces)
        self.fes_interface = ng.FacetFESpace(
            self.mesh, order=order, complex=self.complex
        )

    def _set_grid_function(self):
        """Get grid function on function space."""
        self.gfu = ng.GridFunction(self.fes)

    def get_bilinear_form(self, sigma: ng.CF, interface_admittance: ng.CF):
        U = self.fes.TrialFunction()
        V = self.fes.TestFunction()
        a = ng.BilinearForm(self.fes)

        u = list(U)
        v = list(V)

        # volume terms per color class
        for k, cls in enumerate(self.color_classes):
            if not cls:
                continue
            a += sigma * ng.grad(u[k]) * ng.grad(v[k]) * ng.dx("|".join(cls))

        # interface (two-sided) terms
        for boundary, A, B in self._iter_true_interfaces():
            cA, cB = self.mat2color[A], self.mat2color[B]
            a += (
                interface_admittance
                * (u[cA] - u[cB])
                * (v[cA] - v[cB])
                * ng.ds(boundary)
            )

        # single-sided boundaries (e.g. bottoms): no TMP term here if TMP=0 by design
        # If you need electrode Robin/Neumann, add them here explicitly.

        f = ng.LinearForm(self.fes)
        return a, f

    def compute_error(self, analytical_solution, grad_analytical_solution):
        """Compute the error in L2 and energy (H1) norm."""
        raise NotImplementedError(
            "The error computation for multi-shelled structures "
            "has not yet been implemented."
        )

    def compute_jump_error(self, jump_analytical: ng.CF):
        """Compute the error of the interface jump."""
        raise NotImplementedError(
            "The error computation for multi-shelled structures "
            "has not yet been implemented."
        )

    def compute_impedance(
        self,
        sigma: ng.CF,
        interface_admittance: ng.CF,
        voltage_drop: float = 1.0,
        verbose: bool = False,
    ):
        # bulk
        gradu_tmp = {
            dom: -ng.grad(self.gfu.components[self.mat2color[dom]])
            for dom in self.domains
        }
        gradu = self.mesh.MaterialCF(gradu_tmp)
        Y = ng.Integrate(sigma * gradu * ng.Conj(gradu), self.mesh)
        if verbose:
            print("Bulk admittance:", Y)

        # two-sided interfaces
        for boundary, A, B in self._iter_true_interfaces():
            TMP = self.get_TMP(boundary)  # (u_out - u_in)|Γ
            bnd_Y = ng.Integrate(
                interface_admittance * TMP * ng.Conj(TMP),
                self.mesh,
                definedon=self.mesh.Boundaries(boundary),
            )
            Y += bnd_Y
            if verbose:
                print(f"[{boundary}] {A}↔{B} : adm={bnd_Y}")

        # single-sided interfaces (e.g., bottoms): TMP = 0 (as per your old code)
        for boundary, A in self._iter_single_sided():
            gf_zero = ng.GridFunction(self.fes_interface)
            gf_zero.Set(0, definedon=self.mesh.Boundaries(boundary))
            bnd_Y = ng.Integrate(
                interface_admittance * gf_zero * ng.Conj(gf_zero),
                self.mesh,
                definedon=self.mesh.Boundaries(boundary),
            )
            Y += bnd_Y
            if verbose:
                print(f"[{boundary}] single-sided on {A}: adm={bnd_Y}")

        if verbose:
            print("Final admittance:", Y)
        return voltage_drop**2 / Y

    def export_VTK(
        self,
        sigma_dict: dict,
        filename: str,
        subdivision: int = 0,
        floatsize: str = "double",
        save_field: bool = False,
        save_current_density: bool = False,
        verbose: bool = False,
        legacy=True,
    ):
        # --- build per-material coefficient functions by pulling from the right COLOR component
        solution_cf = {}
        field_cf = {} if (save_field or save_current_density) else None
        current_cf = {} if save_current_density else None

        # (Optional) numeric material ID map for thresholding in ParaView
        non_ecm = [d for d in self.domains if d != "ecm"]
        sorted_domains = non_ecm + (["ecm"] if "ecm" in self.domains else [])
        mat_id = {dom: i for i, dom in enumerate(sorted_domains)}
        cf_mat_id = self.mesh.MaterialCF(mat_id, default=-1)

        for dom in self.domains:
            k = self.mat2color[dom]  # <-- color index
            u_dom = self.gfu.components[k]  # potential on dom (its color’s comp)
            solution_cf[dom] = u_dom

            if save_field or save_current_density:
                E_dom = -ng.grad(self.gfu.components[k])
                if save_field:
                    field_cf[dom] = E_dom
                if save_current_density:
                    sigma = sigma_dict[dom]
                    if verbose:
                        print(f"domain: {dom}, sigma: {sigma}")
                    # J = sigma * E
                    if current_cf is not None:
                        current_cf[dom] = E_dom * sigma

        # pack fields as MaterialCFs
        u_mat = self.mesh.MaterialCF(solution_cf)
        coefs = [cf_mat_id, u_mat.real, u_mat.imag]
        names = ["mat_name", "potential_real", "potential_imag"]

        if save_field:
            E_mat = self.mesh.MaterialCF(field_cf)
            coefs += [E_mat.real, E_mat.imag]
            names += ["field_real", "field_imag"]

        if save_current_density:
            J_mat = self.mesh.MaterialCF(current_cf)
            coefs += [J_mat.real, J_mat.imag]
            names += ["current_real", "current_imag"]

        print(
            f"ParaView threshold: material index in [{min(mat_id.values())}, {max(mat_id.values())}] (ecm at {mat_id.get('ecm', 'n/a')})"
        )

        vtk = ng.VTKOutput(
            ma=self.mesh,
            coefs=coefs,
            names=names,
            filename=filename,
            floatsize=floatsize,
            subdivision=subdivision,
            legacy=legacy,
        )
        vtk.Do()

    def export_surface_jump_VTK(
        self, filename: str, floatsize="double", subdivision: int = 0
    ):
        interface_id = {}
        TMP_acc = ng.GridFunction(self.fes_interface)
        TMP_tmp = ng.GridFunction(self.fes_interface)

        i = 1
        # two-sided: add jump
        for boundary, A, B in self._iter_true_interfaces():
            TMP_tmp.vec.data *= 0.0
            jump_cf = self.get_TMP(boundary)  # u_out - u_in
            TMP_tmp.Set(jump_cf, dual=True, definedon=self.mesh.Boundaries(boundary))
            TMP_acc.vec.data += TMP_tmp.vec.data
            interface_id[boundary] = i
            i += 1

        # single-sided: set zero (as before)
        for boundary, A in self._iter_single_sided():
            TMP_tmp.vec.data *= 0.0
            TMP_tmp.Set(0, definedon=self.mesh.Boundaries(boundary))
            TMP_acc.vec.data += TMP_tmp.vec.data
            interface_id[boundary] = i
            i += 1

        print(f"ParaView threshold: interface id in [1, {i - 1}]")

        surface_TMP = ng.CoefficientFunction(TMP_acc)
        cf_iface_id = self.mesh.BoundaryCF(interface_id, default=-1)
        vtk = ng.VTKOutput(
            ma=self.mesh,
            coefs=[cf_iface_id, surface_TMP.real, surface_TMP.imag],
            names=["interface_name", "TMP_real", "TMP_imag"],
            filename=filename,
            floatsize=floatsize,
            subdivision=subdivision,
        )
        vtk.Do(vb=ng.BND)

    def get_TMP(self, boundary: str) -> ng.GridFunction:
        if self._gf1 is None:
            self._gf1 = ng.GridFunction(self.fes_interface)
        if self._gf2 is None:
            self._gf2 = ng.GridFunction(self.fes_interface)

        A, B = self.outer_inner_pairs[boundary]  # guaranteed tuple here
        cA = self.mat2color[A]
        cB = self.mat2color[B]
        bset = self.mesh.Boundaries(boundary)
        self._gf1.Set(self.gfu.components[cA], definedon=bset)  # u_out (A)
        self._gf2.Set(self.gfu.components[cB], definedon=bset)  # u_in  (B)
        return self._gf1 - self._gf2

    def get_solution_cf(self):
        solution_dict = {}
        for dom in self.domains:
            k = self.mat2color[dom]  # color index
            solution_dict[dom] = self.gfu.components[k]
        return self.mesh.MaterialCF(solution_dict)
