
import numpy as np
import matplotlib.pyplot as plt

k=9
mu0=5 #rest friction
muv=2 #moving frictionduring 20 min
g=1 #gravity
m=1 #mass of boxes
deltat=0.001 #timestep
def mu(x,v): #returns list
   delx1=x[0]-x[1]
   delx2=x[1]-x[2]
   mu_net=np.array([0.0,0.0,0.0])
   
   if abs((delx1)*k/(m*g))<=mu0 and v[0]==0:
      mu_net[0]=((delx1)*k/(m*g))
   else:
      mu_net[0]=np.sign(v[0])*muv
   
   if abs((delx2-delx1)*k/(m*g))<=mu0 and v[1]==0:
      print()
      mu_net[1]=(delx2-delx1)*k/(m*g)
   else:
      mu_net[1]=np.sign(v[1])*muv
   return mu_net

def acc(x,v): #returns 2 accelerations - one for each box (in a list). x, v vectors
   mu_n=mu(x,v)
   delx1=x[0]-x[1]
   delx2=x[1]-x[2]
  
   a=np.array([0.0,0.0,0.0])
   a[0]=round(-k*(delx1)/m+mu_n[0]*g,10)
   a[1]=round(-k*(delx2-delx1)/m+mu_n[1]*g,10)
   a[2]=0
   return a

def eulerx(x,v,h=deltat): #Runga kutta?
   x=x+v*h
   return x

def eulerv(v,a,h=deltat): #Runga kutta?
   v=v+a*h
   return v

x=np.array([0,0,0]) #initial positions
v=np.array([0,0,1]) #initial velocity
time=10
xlist=[]
x0list=[]
vlist=[]
for i in range(int(time/deltat)):

   xlist.append(x[1])
   x0list.append(x[2])
   a=acc(x,v)
   x=eulerx(x,v)
   v=eulerv(v,a)

   
   
tlist=np.linspace(0, time, int(time/deltat))
plt.plot(tlist,xlist, "b-")
plt.plot(tlist,x0list, "r-")

   
   
   
