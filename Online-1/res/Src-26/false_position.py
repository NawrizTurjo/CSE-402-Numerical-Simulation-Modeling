coeffs = []

istr = input("Enter the coeffs: \n")
istr = istr.split()
istr = list(map(int, istr))

coeffs = istr.copy()
print(coeffs)

def fn_eval(coeffs, fn_in):
    fn_out = 0
    for i in range(-1, -len(coeffs)-1, -1):
        fn_out += coeffs[i]*pow(fn_in, -i-1)
    return fn_out


def false_position(coeffs, lower, upper, r_where, flag):
    if (fn_eval(coeffs, lower)) == 0:
        return lower
    elif (fn_eval(coeffs, upper) == 0):
        return upper
    
    elif fn_eval(coeffs, lower) * fn_eval(coeffs, upper) > 0:
        print("choose diff start points")
        return float('inf')
    
    else:
        x_r = lower + fn_eval(coeffs, lower) * (upper - lower) / (fn_eval(coeffs, lower) - fn_eval(coeffs, upper))
        prev_x_r = lower if r_where == 0 else upper
        e_a = float('inf')
        if flag: 
            e_a = 100 * abs((x_r - prev_x_r) / x_r)


        if fn_eval(coeffs, lower) * fn_eval(coeffs, x_r) < 0:
            if (e_a <= 0.5):
                return x_r
            return false_position(coeffs, lower, x_r, 1, True)
        elif fn_eval(coeffs, x_r) * fn_eval(coeffs, upper) < 0:
            if (e_a <= 0.5):
                return x_r
            return false_position(coeffs, x_r, upper, 0, True)
        else:
            return x_r


print(false_position(coeffs, -4, -2.5, 0, False))