
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

k=10
mu0=100 #rest friction
muv=1 #moving friction
g=1 #gravity
m=1 #mass of boxes
deltat=0.01 #timestep
epsilon=1e-15
time=1000
n=int(time/deltat)

state0=1
x0=1
v0=1


eps=1 
xi=1
gamma=1

def f(x): #vector in x = ((state,delta,v))
   state = x[0]
   delta = x[1]
   v = x[2]
   
   stateDOT = -v*(state+(1+eps)*np.log(v))
   deltaDOT = v-1
   vDOT = -gamma**2*(delta+(1/xi)*(state+np.log(v)))

   return np.array([stateDOT, deltaDOT, vDOT])
   

def rk4(x0, h, n):
    x = np.zeros((n+1,3)) #creating np.array where we save our data

    x[0] = x0 #set initial conditions

    t=np.linspace(0,h*n,n+1)
    
    for i in range(1, n + 1):
       
       xi=x[i-1]
       k1 = h * f(xi)
       k2 = h * f(xi + 0.5 * k1)
       k3 = h * f(xi + 0.5 * k2)
       k4 = h * f(xi + k3)
       x[i] = xi + (k1 + 2*k2+ 2*k3 + k4) / 6
    return x,t
   
x_ini=np.array([1,1,1]) #initial positions,velocity
all_data=rk4(x_ini,deltat,n)
#all_data_list=all_data.tolist()
timelist=all_data[1]
zero = np.array([[float(0)]*(n+1)]*2)

stateEVO = [x[0] for x in all_data[0]]
deltaEVO = [x[1] for x in all_data[0]]
vEVO = [x[2] for x in all_data[0]]
   
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
   
ax.plot( xs=stateEVO, ys=deltaEVO, zs=vEVO)
ax.set_xlabel('state')
ax.set_ylabel('delta')
ax.set_zlabel('velocity')

fig2 = plt.figure()
plt.plot(timelist, stateEVO, color="green", label="state")
plt.plot(timelist, deltaEVO, color="blue", label="delta")
plt.plot(timelist, vEVO, color="red", label="velocity")
plt.legend()

