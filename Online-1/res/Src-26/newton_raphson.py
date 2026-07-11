def fn_eval(coeffs, fn_in):
    res = 0
    for coeff in coeffs:
        res = res*fn_in + coeff
    return res

def fn_diff_eval(coeffs, fn_in, thres=1e-3):
    return (fn_eval(coeffs, fn_in+thres) - fn_eval(coeffs, fn_in))/thres

def newton_raphson(coeffs, start, e_s):
    if (fn_eval(coeffs, start) == 0):
        return start
    
    e_a = float('inf')
    x_prev = start
    x = start
    while e_a > e_s:
        x = x_prev - fn_eval(coeffs, x)/fn_diff_eval(coeffs, x)
        e_a = 100 * abs((x - x_prev)/x)
        print(x, e_a)
        x_prev = x

    return x



istr = input("Enter the coeffs (highest degree first): \n")
coeffs = list(map(int, istr.split()))
print("Coefficients:", coeffs)
root = newton_raphson(coeffs, -20, 0.5)
if root != float('inf'):
    print(f"Root found: {root:.6f}")