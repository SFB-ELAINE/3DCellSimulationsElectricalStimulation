# Validation simulations

## Convergence rates

We prove that the numerical implementation in NGSolve
has the expected optimal order of convergence.

For that, we used a 2D example on a canonical geometry,
a 3D example of a cell exposed to an external electric field.

## Impedance spectrum

We compute the impedance spectrum of randomly distributed cells
in suspension in a parallel plate capacitor.
For this configuration, an analytical expression is known.

## LinKK test
We demonstrate that the LinnKK test can be used to check whether impedance data satisfy the Kramers-Kronig relations.
When we deliberately omitted the impedance contribution of the cell membrane, the Lin-KK test correctly detected this mistake.
It can also be employed to check the correctness of the computed impedance with respect to the convergence of the solver.
