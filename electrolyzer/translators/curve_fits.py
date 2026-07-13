def cubic_with_sqrt_5coeffs(pwr, p1, p2, p3, p4, p5, p6):
    i_stack = p1 * (pwr**3) + p2 * (pwr**2) + (p3 * pwr) + (p4 * pwr ** (1 / 2)) + p5
    return i_stack
