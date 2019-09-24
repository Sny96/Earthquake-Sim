
import numpy as np
import matplotlib.pyplot as plt

k=10
mu0=100 #rest friction
muv=1 #moving friction
g=1 #gravity
m=1 #mass of boxes
deltat=0.01 #timestep
epsilon=1e-15
time=150
n=int(time/deltat)
def mu(x,v): #returns list
   delx1=x[0]-x[1]
   delx2=x[1]-x[2]
   mu_net=np.array([0.0,0.0,0.0])
   
   if abs((delx1)*k/(m*g))<=mu0 and v[0]<=epsilon:
      mu_net[0]=((delx1)*k/(m*g))
   else:
      mu_net[0]=-np.sign(v[0])*muv
   
   if abs((delx2-delx1)*k/(m*g))<=mu0 and v[1]<=epsilon:
      mu_net[1]=(delx2-delx1)*k/(m*g)
   else:
      mu_net[1]=-np.sign(v[1])*muv
   return mu_net

def acc(x,v): #returns 2 accelerations - one for each box (in a list). x, v vectors
   mu_n=mu(x,v)
   delx1=x[0]-x[1]
   delx2=x[1]-x[2]
  
   a=np.array([0.0,0.0,0.0])
   a[0]=-k*(delx1)/m+mu_n[0]*g
   if abs(a[0])<=epsilon:
      a[0]=0
   a[1]=-k*(delx2-delx1)/m+mu_n[1]*g
   if abs(a[1])<=epsilon:
      a[1]=0
   a[2]=0
  # print(-k*(delx2-delx1)/m,mu_n[1]*g)
   
   return a

def f(x): #x is a vector of x,v
   return np.array([x[1], acc(x[0],x[1])])
   

def rk4(x0, h, n):
    x = np.zeros((n+1,2,3))

    x[0] = x0 #set initial conditions

    t=np.linspace(0,h*n,n+1)
    for i in range(1, n + 1):
       
       xi=x[i-1]
       print(f(xi))
       k1 = h * f(xi)
       k2 = h * f(xi + 0.5 * k1)
       k3 = h * f(xi + 0.5 * k2)
       k4 = h * f(xi + k3)
       x[i] = xi + (k1 + 2*k2+ 2*k3 + k4) / 6
    return x,t
   
x_ini=np.array([[0,0,0.1],[0,0,1]]) #initial positions,velocity
all_data=rk4(x_ini,deltat,n)
#all_data_list=all_data.tolist()
timelist=all_data[1]
zero = np.array([[float(0)]*(n+1)]*2)
box_1_rel = np.copy(zero) 
box_2_rel = np.copy(zero) 

box_1 = np.copy(zero) 
box_2 = np.copy(zero) 
box_3 = np.copy(zero) 
print(np.shape(box_1))
for i,data in enumerate(all_data[0]):
   box_1[0][i] = data[0][0]
   box_1[1][i] = data[1][0]
   box_2[0][i] = data[0][1]
   box_2[1][i] = data[1][1]
   box_3[0][i] = data[0][2]
   box_3[1][i] = data[1][2]
   box_1_rel[0][i] = data[0][0]-box_3[0][i]
   box_1_rel[1][i] = data[1][0]-box_3[1][i]
   box_2_rel[0][i] = data[0][1]-box_3[0][i]
   box_2_rel[1][i] = data[1][1]-box_3[1][i]
   
   

plt.plot(timelist, box_2[0])
plt.plot(timelist, box_1[0])
#plt.plot(timelist, -box_3[0])

   
   
   
