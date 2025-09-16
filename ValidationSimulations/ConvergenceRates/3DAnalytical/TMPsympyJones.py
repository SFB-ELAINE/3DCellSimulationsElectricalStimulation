import sympy


def return_solution(cartesian=False):
    # define variables
    r, theta = sympy.symbols("r, theta")

    R, dm, ki, km, ke, E = sympy.symbols("R, dm, ki, km, ke, E")
    A, K = sympy.symbols("A, K")

    # define solution
    Phii = -K * r * sympy.cos(theta)
    Phie = (-E * r + A / r**2) * sympy.cos(theta)

    # define equations based on different conditions
    Eqns = [
        km / dm * (Phii.subs(r, R) - Phie.subs(r, R)) + ki * Phii.diff(r).subs(r, R),
        ke * Phie.diff(r).subs(r, R) - ki * Phii.diff(r).subs(r, R),
    ]
    res = sympy.solve(Eqns, [A, K])

    # compute TMP
    TMP = Phii.subs(K, res[K]).subs(r, R) - Phie.subs(A, res[A]).subs(r, R)

    # reference solution
    U = (2 * km + ki) * (km - ke) * R**3 + (ki - km) * (2 * km + ke) * R**3
    U = U / ((2 * km + ki) * (2 * ke + km) * R**3 + 2 * (ki - km) * (km - ke) * R**3)

    # compute U from sympy result
    # sympy_res = (res[A] / (E * R**3))
    # assert (U - sympy_res).factor() == 0, "result from Sukhorukov paper must match!"

    # reference solution for TMP
    ref_TMP = 3 * E * R * sympy.cos(theta) / 2
    assert (TMP.subs(km, 0) - ref_TMP).factor() == 0, (
        "Attention: result for non-conductive membrane must match!"
    )

    # Solution inside cell
    Phii_sol = Phii.subs(K, res[K])

    # Solution outside cell
    Phie_sol = Phie.subs(A, res[A])
    if cartesian:
        x, y, z = sympy.symbols("x, y, z")
        Phii_cart = Phii_sol.subs(
            [
                (r, sympy.sqrt(x**2 + y**2 + z**2)),
                (theta, sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))),
            ]
        )
        Phie_cart = Phie_sol.subs(
            [
                (r, sympy.sqrt(x**2 + y**2 + z**2)),
                (theta, sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))),
            ]
        )
        TMP_cart = TMP.subs(
            [
                (r, sympy.sqrt(x**2 + y**2 + z**2)),
                (theta, sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))),
            ]
        )
        # return sympy.simplify(Phii_cart), sympy.simplify(Phie_cart)
        return Phii_cart, Phie_cart, TMP_cart
    else:
        return Phii_sol, Phie_sol, TMP


def error_at_boundary(factor):
    """
    consider rectangular domain with fixed value at boundary.
    This introduces an error.
    The edge length of the rectangular domain is a certain factor
    longer than the cell radius.
    Returns a sympy expression.
    """
    inner, outer = return_solution(cartesian=True)
    U, d, E, x, y, z = sympy.symbols("U, d, E, x, y, z")
    dirichlet = -U / 2
    tmp = outer.subs("E", U / d)
    res = tmp.subs([("x", 0), ("y", 0), ("z", d / 2)])
    return sympy.Abs(dirichlet - res.subs("R", d / int(factor)))


def value(x=0, y=0, z=0, E=1):
    """
    Value for external field E at position (x, y, z).
    """
    inner, outer, TMP = return_solution(cartesian=True)
    U, d, E = sympy.symbols("U, d, E")
    tmp = outer.subs("E", E)
    res = tmp.subs([("x", x), ("y", x), ("z", z)])
    return res


def main():
    inner, outer = return_solution()
    print("Solution inside cell")
    print(inner.collect("r"))
    print("Solution outside cell")
    print(outer.collect("r"))

    print("Solution at top edge")
    U, d = sympy.symbols("U, d")
    tmp = outer.subs("E", U / d)
    res = tmp.subs([("r", d / 2), ("theta", 0)])

    print(res)
    print("Limit if d >> R")
    print(sympy.limit(res, d, sympy.oo))
    print("Expression if d == 10 R")
    print(res.subs("R", d / 10).collect("U"))
    print("Cartesian solutions:")
    inner, outer = return_solution(cartesian=True)
    print("Solution inside cell")
    print(inner)
    print("Solution outside cell")
    print(outer)

    print("Solution at top edge")
    U, d = sympy.symbols("U, d")
    x, y, z = sympy.symbols("x, y, z")
    tmp = outer.subs("E", U / d)
    res = tmp.subs([("x", 0), ("y", 0), ("z", d / 2)])

    print(res)
    print("Limit if d >> R")
    limit = sympy.limit(res, d, sympy.oo)
    print(limit)
    print("Expression if d == 10 R")
    print(res.subs("R", d / 10).collect("U"))
