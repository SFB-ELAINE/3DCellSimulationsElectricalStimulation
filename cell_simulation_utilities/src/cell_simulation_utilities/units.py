class UnitConverter:
    def __init__(self, length_unit, mass_unit, time_unit, current_unit):
        self.L = length_unit
        self.M = mass_unit
        self.T = time_unit
        self.I = current_unit

    @property
    def conductivity_unit(self) -> float:
        return self.L ** (-3) * self.M ** (-1) * self.T**3 * self.I**2

    @property
    def permittivity_unit(self) -> float:
        return self.L ** (-3) * self.M ** (-1) * self.T**4 * self.I**2

    def field_unit(self) -> float:
        return self.L * self.M * self.T ** (-3) * self.I ** (-1)

    def current_density_unit(self) -> float:
        return self.conductivity_unit * self.field_unit

    def frequency_unit(self) -> float:
        return 1.0 / self.T
