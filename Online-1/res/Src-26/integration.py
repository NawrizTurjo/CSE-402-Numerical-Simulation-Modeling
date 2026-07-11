def fn_eval(coeffs, x):
    res = 0
    for coeff in coeffs:
        res = res*x + coeff
    return res

def integrate(coeffs, l, u, h):
    res = 0
    i = l
    while i < u:
        res = res + fn_eval(coeffs, i+(h/2)) * h
        i += h
    
    return res

print(integrate([1, 0], 0, 5, 0.001))
