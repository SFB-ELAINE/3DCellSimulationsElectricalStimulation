import ngsolve
import numpy as np
from scipy.constants import epsilon_0 as e0


class Conductivity:
    def __init__(self, domains: list, complex=False):
        self.domains = []
        # catch connected domains
        for domain in domains:
            if "|" in domain:
                for dmn in domain.split("|"):
                    self.domains.append(dmn)
            else:
                self.domains.append(domain)
        self.enum_domains = {}
        for i, domain in enumerate(self.domains):
            self.enum_domains[domain] = i

        self.parameters = {}
        self.complex = complex
        for domain in self.domains:
            if self.complex:
                self.parameters[domain] = ngsolve.ParameterC(0.0)
            else:
                self.parameters[domain] = ngsolve.Parameter(0.0)

    def prepare_parameters(
        self,
        conductivities: dict,
        permittivities: dict,
        frequency: float,
        scaling_factor: float = 1.0,
    ):
        if not conductivities.keys() == permittivities.keys():
            raise ValueError(
                "Conductivity and permittivities are not provided for same domains."
            )
        if not set(self.domains).issubset(conductivities.keys()):
            raise ValueError("Need to provide material parameters for all domains.")

        parameters = {}
        for domain in self.domains:
            sigma = (
                conductivities[domain]
                + 1j * 2.0 * np.pi * frequency * permittivities[domain] * e0
            )
            parameters[domain] = scaling_factor * sigma
        return parameters

    def set_parameters(self, parameters: dict):
        for domain in parameters:
            self.parameters[domain].Set(parameters[domain])

    def check_parameters_change(self, parameters: dict, tolerance=0.01):
        real_part_close = []
        imag_part_close = []
        # go through all materials
        for domain in parameters:
            old_value = self.parameters[domain].Get()
            new_value = parameters[domain]
            real_part_close.append(
                np.isclose(
                    old_value.real, new_value.real, rtol=tolerance, atol=tolerance
                )
            )
            if self.complex:
                imag_part_close.append(
                    np.isclose(
                        old_value.imag, new_value.imag, rtol=tolerance, atol=tolerance
                    )
                )
            else:
                imag_part_close.append(True)
        all_parameters_close = all(real_part_close) and all(imag_part_close)
        return not all_parameters_close

    def get_coefficient_function(self, mesh: ngsolve.Mesh):
        return mesh.MaterialCF(self.parameters)

    def export_VTK(
        self,
        mesh: ngsolve.Mesh,
        filename: str,
        subdivision: int = 0,
        floatsize="double",
    ):
        conductivity = self.get_coefficient_function(mesh)
        # to avoid issue with datatype
        gfu = ngsolve.GridFunction(ngsolve.L2(mesh, complex=True))
        gfu.Set(conductivity)
        domains = mesh.MaterialCF(self.enum_domains)
        vtk = ngsolve.VTKOutput(
            ma=mesh,
            coefs=[gfu.real, gfu.imag, domains],
            names=["sigReal", "sigImag", "domain"],
            filename=filename,
            floatsize=floatsize,
            subdivision=subdivision,
        )
        vtk.Do()


class BoundaryAdmittance(Conductivity):
    def get_coefficient_function(self, mesh: ngsolve.Mesh):
        return mesh.BoundaryCF(self.parameters)

    def export_VTK(
        self,
        mesh: ngsolve.Mesh,
        filename: str,
        subdivision: int = 0,
        floatsize="double",
    ):
        raise NotImplementedError(
            "VTK export for BoundaryAdmittance not yet implemented"
        )
