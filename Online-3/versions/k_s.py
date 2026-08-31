import numpy as np

data=[0.44,0.81,0.14,0.05,0.93]
data=sorted(data)

N=5
alpha=0.05
dplus=0
dminus=0


for i in range(N):
    dplus_temp=(i+1)/N-data[i]
    dminus_temp=data[i]-i/N
    dplus=max(dplus,dplus_temp)
    dminus=max(dminus,dminus_temp)

d=max(dplus,dminus)
if d > 0.565:
    decision = "Reject H0"
else:
    decision = "Do not reject H0"
print("Decision =", decision)
