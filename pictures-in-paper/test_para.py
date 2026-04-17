delta=[1.3, 0.74, 0.3]
S=[1.3,0.65,0.24,0]
a=[S[i+1]-delta[i] for i in range(3)]
b=[S[i+1]/delta[i] for i in range(3)]
print(a)
print(b)
print((S[2]*delta[0]-S[1]*delta[1])/(S[1]+delta[1]-S[2]-delta[0]))
print((S[3]*delta[1]-S[2]*delta[2])/(S[2]+delta[2]-S[3]-delta[1]))
print((S[3]*delta[0]-S[1]*delta[2])/(S[1]+delta[2]-S[3]-delta[0]))