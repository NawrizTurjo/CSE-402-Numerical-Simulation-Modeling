# let's take an eqn as input of coeff's of a polynomial
coeffs = []
istr = input("Enter the coefficients, in descending power.\n")
istr = istr.split(" ")
istr = list(map(int, istr))
coeffs = istr.copy()
print(coeffs)

degree = len(istr)-1

fn_in = int(input("x = "))


def fn_eval(coeffs, fn_in):
    fn_out = 0
    for i in range(-1, -len(coeffs)-1, -1):
        fn_out += coeffs[i]*pow(fn_in, -i-1)
    return fn_out


iter = 0 
def bisection(coeffs, lower, upper, mid_where): #mid_where = 0 lower, 1 upper
    # global iter
    # iter = iter+1
    # if(iter > 10000):
    #     return upper if mid_where==1 else lower

    if fn_eval(coeffs, lower)*fn_eval(coeffs, upper) == 0:
        return lower if fn_eval(coeffs, lower) == 0 else upper
    elif fn_eval(coeffs, lower) * fn_eval(coeffs, upper) < 0:
        mid = (lower+upper)/2
        if fn_eval(coeffs, mid) == 0:
            return mid
        elif fn_eval(coeffs, lower) * fn_eval(coeffs, mid) < 0:
            prev_mid = upper if mid_where == 1 else lower
            e_a = 100 * abs((mid - prev_mid) / mid)
            print(prev_mid, mid)
            if (e_a <= .5): return mid
            return bisection(coeffs, lower, mid, 1)
        else:
            prev_mid = upper if mid_where == 1 else lower
            e_a = 100 * abs((mid - prev_mid) / mid)
            print(prev_mid, mid)
            if (e_a <= .5): return mid
            return bisection(coeffs, mid, upper, 0)

    else:
        print("choose diff starting points")
        return float('inf')
    
print(bisection(coeffs, -4, -2.5, 0))