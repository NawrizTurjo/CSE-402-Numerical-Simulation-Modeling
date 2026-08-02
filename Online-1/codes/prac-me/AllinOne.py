import numpy as np
import matplotlib.pyplot as plt
import math
def func(x):
    return np.power(x,3)-6*np.power(x,2)+11*np.power(x,1)-6.2
def derivative(x,h=0.00001):
    return (func(x+h)-func(x))/h
def plot(st,en):
    X=np.linspace(st,en,500)
    Y=func(X)
    plt.plot(X,Y)
    plt.show()
    # plt.yscale('log')
def calc_sig_digit(err): 
    if err==0:
        return 9999
    return math.floor(2-math.log10(2*err))
def calc_error(true,approx):
    return abs((true-approx)/true)*100
def bisection(x_l,x_u,thres):
    iter=0
    error=10000
    prev_x_m=10000
    sig_digit=-1
    upper_change=0
    lower_change=0
    while(1):
        iter=iter+1
        x_m=(x_l+x_u)/2
        f_x_l=func(x_l)
        f_x_m=func(x_m)
        if(iter!=1):
            error=calc_error(x_m,prev_x_m)
            sig_digit=calc_sig_digit(error)
        prev_x_m=x_m
        if(iter!=1):
            print(f"Iter:{iter} X_l:{x_l:.4f} X_u:{x_u:.4f} X_m:{x_m:.4f} F(x_m):{f_x_m:.4f} Error:{error:.4f} SigDigit:{sig_digit:.4f}")
        else:
            print(f"Iter:{iter} X_l:{x_l:.4f} X_u:{x_u:.4f} X_m:{x_m:.4f} F(x_m):{f_x_m:.4f} Error:------ SigDigit:{sig_digit:.4f}")
        val=f_x_l*f_x_m
        if(val<0):
            x_u=x_m
            upper_change=upper_change+1
        elif(val>0):
            x_l=x_m
            lower_change=lower_change+1
        else:
            break
        if(sig_digit>=thres):
            break
    return iter,x_m,error,sig_digit,upper_change,lower_change
def falseposition(x_l,x_u,thres):
    iter=0
    error=10000
    prev_x_m=10000
    sig_digit=-1
    upper_change=0
    lower_change=0
    while(1):
        iter=iter+1
        
        
        f_x_l=func(x_l)
        
        f_x_u=func(x_u)
        x_m=x_u-(f_x_u*(x_u-x_l))/(f_x_u-f_x_l)
        f_x_m=func(x_m)
        if(iter!=1):
            error=calc_error(x_m,prev_x_m)
            sig_digit=calc_sig_digit(error)
        prev_x_m=x_m
        if(iter!=1):
            print(f"Iter:{iter} X_l:{x_l:.4f} X_u:{x_u:.4f} X_m:{x_m:.4f} F(x_m):{f_x_m:.4f} Error:{error:.4f} SigDigit:{sig_digit:.4f}")
        else:
            print(f"Iter:{iter} X_l:{x_l:.4f} X_u:{x_u:.4f} X_m:{x_m:.4f} F(x_m):{f_x_m:.4f} Error:------ SigDigit:{sig_digit:.4f}")
        val=f_x_l*f_x_m
        if(val<0):
            x_u=x_m
            upper_change=upper_change+1
        elif(val>0):
            x_l=x_m
            lower_change=lower_change+1
        else:
            break
        if(sig_digit>=thres):
            break
    return iter,x_m,error,sig_digit,upper_change,lower_change

def newton_rapson(x_i,thres):
    iter=0
    error=10000
    sig_digit=-1
    while(1):
        iter=iter+1
        f_x_i=func(x_i)
        der_f_x_i=derivative(x_i)
        x_i1=x_i-(f_x_i/der_f_x_i)
        error=calc_error(x_i1,x_i)
        sig_digit=calc_sig_digit(error)
        print(f"Iter:{iter} X_i:{x_i:.4f} X_(i+1):{x_i1:.4f} f(X_i):{f_x_i:.4f} F'(x_i):{der_f_x_i:.4f} F(x_(i+1)):{func(x_i1):.4f} Error:{error:.4f}SigDigit:{sig_digit:.4f}")
        x_i=x_i1
        if(sig_digit>=thres):
            break
        
    return iter,x_i,error,sig_digit
def findSignChangeInRange(st,en,h=0.3):
    x=st
    while(x<en):
        brac_1=x
        brac_2=x+h
        val_1=func(brac_1)
        val_2=func(brac_2)
        x=x+h
        if(val_1*val_2<0):
            print("Sign Change detected Can start")
            print("Bisection----------------------------------")
            iter,x_m,error,sig_digit,upper_change,lower_change=bisection(brac_1,brac_2,4)
            print(f"Iter:{iter} Root:{x_m:.4f} Error:{error:.4f} SigDigit:{sig_digit:.4f} UpperChange:{upper_change} LowerChange:{lower_change}")
            print("Bisection Ended------------------------------")
            print("False Position----------------------------------")
            iter,x_m,error,sig_digit,upper_change,lower_change=falseposition(brac_1,brac_2,4)
            print(f"Iter:{iter} Root:{x_m:.4f} Error:{error:.4f} SigDigit:{sig_digit:.4f} UpperChange:{upper_change} LowerChange:{lower_change}")
            print("False Position Ended------------------------------")
    print("Newton Rapson-----------------------")
    iter,x_i,error,sig_digit=newton_rapson(st,4)
    print(f"Iter:{iter} Root:{x_i:.4f} Error:{error:.4f} SigDigit:{sig_digit:.4f}")

findSignChangeInRange(0,4)
plot(0,4)
