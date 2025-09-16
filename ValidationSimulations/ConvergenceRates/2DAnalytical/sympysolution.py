import sympy


def return_solution():
    # define variables
    x, y = sympy.symbols("x, y")

    R, ki, ke = sympy.symbols("R, ki, ke")
    a, b, c = sympy.symbols("a, b, c")

    # define solution
    soli = a * (x**2 - y**2)  # inside
    sole = (x**2 - y**2) * (b + c / (x**2 + y**2) ** 2)  # outside

    # define equations based on different conditions
    Eqns = [
        (sole - 4 / 9 * (x**2 - y**2)).subs([(x, 0), (y, 3 / 2)]),  # Dirichlet BC
        (ki * soli.diff(y) - ke * sole.diff(y)).subs(
            [(x, 0), (y, 1)]
        ),  # continuity of normal derivative
        (R * ki * soli.diff(y) - (sole - soli)).subs([(x, 0), (y, 1)]),
    ]  # contact resistance
    res = sympy.solve(Eqns, [a, b, c])
    soli = soli.subs(a, res[a])
    sole = sole.subs([(b, res[b]), (c, res[c])])
    difference = soli - sole
    return soli, sole, difference
