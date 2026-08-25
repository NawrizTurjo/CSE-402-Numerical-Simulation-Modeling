#10:39 pm - 
import math 
def gen(seed,a,c,m,n):
    X = seed 
    X_s, R_s = [], [] 
    for _ in range(n):
        X = (a*X+c)%m
        X_s.append(X)
        R_s.append(X/m)
    return X_s,R_s
 
def gen(seed, a, c, m, n):
   X = seed
   X_s, R_s = [], []
   for _ in range(n):
      X = (a*X + c) % m
      X_s.append(X)
      R_s.append(X/m)
   return X_s, R_s

def isPrime(number):
   sqrt_n = int(math.sqrt(number))
   for i in range(2,sqrt_n+1):
      if number%i==0:
         return False 
   return True 


def is_prime(n):
   if n < 2:
      return False
   for i in range(2, int(math.sqrt(n) + 1)):
      if n % i == 0:
         return False
      
   return True

def find_period(seed,a,c,m):
   #  b = math.log2(m)
   #  print(f"im {b}")
    if(m & (m-1) == 0):
       if(c==0):
          if(seed%2!=0 and (((a-3)%8==0)or((a-5)%8==0))):
             print(f"Maximum Period is {m//4}")
            
       elif(math.gcd(c,m)==1 and (a-1)%4==0):
          print(f"Maximum period is {m}")

    elif(c==0 and isPrime(m)):
       if a % m == 0:
         pass
       else:
         k = 1 
         val = a%m  
         while(val!=1):
            val = (val*a)%m  
            k = k+1 
         if(k==m-1):
            print(f"Maximum period is {m-1}")
          
    seen = {}
    X = seed 
    X_s = []
    i = 0 
    while(X not in seen):
     seen[X] = i
     #print(seen)
     X_s.append(X)
     X = (a*X+c)%m 
     i += 1 
    #print([seen])
    return i-seen[X],X_s

import math



def ks_test(sample):
   N = len(sample)
   s = sorted(sample)
   D_plus = max([i/N - s[i-1] for i in range(1,N+1)])
   D_minus = max([s[i-1]-(i-1)/N for i in range(1,N+1)])
   D = max(D_plus,D_minus)
   return D,D_plus,D_minus





def chi_square(sample,bins):
   N = len(sample)
   E = N / bins 
   chi = 0
   counts = [0]*bins 
   for i in sample:
      idx = min(bins-1,int(math.floor(i*bins)))
      counts[idx]+= 1
   print(counts)
   for i in range(bins): 
    chi += ((counts[i]-E)**2)/E

   return chi 

# def auto_correlation_test(R,i,l,N,alpha=0.05):
#    # i + (M+1)*l = N
#    M = (N - i)//l - 2 
#    print(M)
#    # idx = [i-1 + k*l for k in range(M+1)]
#    idx=[]
#    products=0
#    while i+l<=N:
#       idx.append(i)
#       i=i+l
#       # products+=R[i]*R[i+l]
   
#    summ = 0 
#    print(idx)
   
#    for j in range(len(idx)-1):
#       summ += R[idx[j]] * R[idx[j+1]]
#    print(summ)
#    rho_hat = (1/(M+1))*summ - 0.25   
#    sigma = math.sqrt(13*M+7)/(12*(M+1))
#    Z0 = rho_hat/sigma 
#    return rho_hat,sigma,Z0 
def auto_correlation_test(R,i,l,N,alpha=0.05):
   # i + (M+1)*l = N
   M = (N - i)//l - 1 
   print(M)
   idx = [i-1 + k*l for k in range(M+2)]
   #idx=[]
   products=0
   # while i+l<=N:
   #    idx.append(i)
   #    i=i+l
   #    # products+=R[i]*R[i+l]
   
   summ = 0 
   print(idx)
   
   for j in range(len(idx)-1):
      summ += R[idx[j]] * R[idx[j+1]]
   print(summ)
   rho_hat = (1/(M+1))*summ - 0.25   
   sigma = math.sqrt(13*M+7)/(12*(M+1))
   Z0 = rho_hat/sigma 
   return rho_hat,sigma,Z0 






import random 
def monte_carlo(g,a,b,n,seed):
   summ = 0
   random.seed(seed)
   numbers = [random.uniform(a,b) for i in range(n)]
   for i in numbers:
      summ += g(i)
   avg = summ/n 
   return avg * (b-a) 

def monte_carlo_estimate_pi(n):
   hits = 0 
   for _ in range(n):
      x,y = random.random(),random.random()
      #print(x,y)
      if x**2 + y**2 <=1: 
         hits+=1 
   return 4 * hits/n 

Xs, Rs = gen(seed=27, a=17, c=43, m=100, n=4)
#print(Xs)  # [2, 77, 52, 27]
#print(Rs)  # [0.02, 0.77, 0.52, 0.27] 

#print(find_period(13,11,0,64))

# # 1) Multiplicative, m power of 2 (slide's Example 2) -> "Maximum Period is 16", empirical period 16
# print(find_period(seed=1, a=13, c=0, m=64))

# # 2) Mixed, m power of 2, satisfies Hull-Dobell -> "Maximum period is 64", empirical period 64
# print(find_period(seed=5, a=5, c=1, m=64))

# # 3) Multiplicative, m prime, a IS a primitive root -> "Maximum period is 30", empirical period 30
# print(find_period(seed=1, a=3, c=0, m=31))

# # 4) Multiplicative, m prime, a is NOT a primitive root -> no theory line printed, empirical period 5
# print(find_period(seed=1, a=2, c=0, m=31))

# # 5) Mixed, m NOT a power of 2 -- your own LCM textbook example!
# #    No theory line printed (gap #3 above), empirical period 4
# print(find_period(seed=27, a=17, c=43, m=100))

# 6) DO NOT RUN as-is before applying the guard fix -- infinite loop (bug #2)
# print(find_period(seed=1, a=31, c=0, m=31))#
# after the fix: prints nothing (order undefined), empirical result:
#print(find_period(seed=1, a=31, c=0, m=31))  # safe only with the fixed version above


sample = [0.44, 0.81, 0.14, 0.05, 0.93]
#print(ks_test(sample))
sample = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,0.05]
#print(chi_square(sample,10))
print(auto_correlation_test([0.23, 0.28, 0.33, 0.27, 0.05, 0.36], 1, 1, 6))
#print(auto_correlation_test([100, 2, 3, 466, 5, 66, 70], 0, 1, 6))
#print(monte_carlo(math.sin,0,math.pi,10000,1))
#print(monte_carlo_estimate_pi(1000))
