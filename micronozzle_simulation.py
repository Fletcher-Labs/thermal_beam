
import numpy as np

import math
from numba import jit
import random
import scipy
from scipy import stats

from scipy.special import erf
from scipy.interpolate import interp1d as interp

from tqdm import tqdm


## Key physical constants
hbar=1.0545718 * 10**-34; #[J/s]
g=9.81; #Acceleration due to gravity [m/s^2]
mu_b=9.274001 * 10**-24; #Bohr magneton [J/T]
k_b=1.38064852 * 10**-23; #Boltzmann constant [J/K]
amu = 1.66054*10**-27; #[kg conversion]

## Er constants
m_er=167.259 * amu; #[kg]
linewidth_er=29.7*10**6; #[Hz]
lambda_er=401*10**-9; #[m]
k_er=2*np.pi/lambda_er;
Isat_er=602; #[W/m^2]
recoilvel_er=hbar*k_er/m_er;
capture_vel_er=15; #[m/s] actual is 5.6 !!!!


## Geometric factors
cruc_diam=0.012; #12mm in m
cruc_len=0.091; #91mm in m

hl_len=0.061; #61mm in m

#Micronozzle array
num_holes=700;
hole_diam=0.0002; #200 micron in m
nozzle_len=0.01; #10mm in m
array_diam=0.006; #6mm in m

ap2_to_exit_len=0.011; #11mmm in m

oven_len_tot=cruc_len+hl_len+ap2_to_exit_len;

## Temperature/operating conditions (Er)
T_ovenc=1200; #[C]]
T_oven=T_ovenc+273.15; #[K]
T_hl=T_oven+100;

num_er_atoms=30000;

###############################################################################

def generate_positions(n,hole_diam,cruc_len):
    """ generate a set of position vectors in 3D from the oven geometry """
    pz=np.random.uniform(cruc_len,cruc_len,n);
    px=np.random.uniform(-hole_diam/2,hole_diam/2,n);
    py=np.random.uniform(-hole_diam/2,hole_diam/2,n);

    return px, py, pz

###############################################################################

## Random velocity generation from a Maxwell-Boltzmann distribution. I follow:
# https://github.com/tpogden/quantum-python-lectures/blob/master/11_Monte-Carlo-Maxwell-Boltzmann-Distributions.ipynb

vmin=0;
vmax=1500; #[m/s]

###############################################################################

def MB_speed(v,m,T):
    """ Maxwell-Boltzmann speed distribution for speeds """
    kB = 1.38e-23
    return (m/(2*np.pi*kB*T))**1.5 * 4*np.pi * v**2 * np.exp(-m*v**2/(2*kB*T))

###############################################################################

def MB_CDF(v,m,T):
    """ Cumulative Distribution function of the Maxwell-Boltzmann speed distribution """
    kB = 1.38e-23
    a = np.sqrt(kB*T/m)
    return erf(v/(np.sqrt(2)*a)) - np.sqrt(2/np.pi)* v* np.exp(-v**2/(2*a**2))/a

###############################################################################

def generate_velocities(n,m,T):
    """ generate a set of velocity vectors in 3D from the MB inverse CDF function """
    # create CDF
    vs = np.arange(0,vmax,0.1)
    cdf = MB_CDF(vs,m,T) # essentially y = f(x)

    #create interpolation function to CDF to find speeds
    inv_cdf = interp(cdf,vs,fill_value="extrapolate")

    rand_nums = np.random.random(n)
    speeds = inv_cdf(rand_nums)
    
    # spherical polar coords - generate random angle for velocity vector, uniformly distributed over the surface of a sphere
    theta = np.arccos(np.random.uniform(-1,1,n))
    phi = np.random.uniform(0,2*np.pi,n)
    
    # convert to cartesian units
    vx = speeds * np.sin(theta) * np.cos(phi) 
    vy = speeds * np.sin(theta) * np.sin(phi)
    vz = speeds * np.cos(theta)
    
    return speeds, vx, vy, vz

###############################################################################

###########################
##### APERTURE UPDATES ####
########################### 

vmin=0;
vmax=1500; #[m/s]

###############################################################################

def MB_speed_ap(v,m,T,alpha):
    """ Maxwell-Boltzmann speed distribution for speeds w/ aperture """
    kB = 1.38e-23
    dist=(m/(2*np.pi*kB*T))**1.5 * 4*np.pi * v**3 * np.exp(-m*v**2/(2*kB*T));
    norm=np.trapz(dist,v);
    res=dist/norm*np.cos(alpha);
    return res

###############################################################################

def MB_CDF_ap(v,m,T):
    """ Cumulative Distribution function of the Maxwell-Boltzmann speed distribution """
    kB = 1.38e-23
    dist=0.797885*np.sqrt(m/kB/T)*(2*kB*T-np.exp(-m*v**2/(2*kB*T))*(2*kB*T+m*v**2))/m;
    norm=max(dist);
    res=dist/norm;
    return res

###############################################################################

def generate_velocities_ap(n,m,T):
    """ generate a set of velocity vectors in 3D from the MB inverse CDF function """
    # create CDF
    vs = np.arange(0,vmax,0.1)
    cdf = MB_CDF_ap(vs,m,T) # essentially y = f(x)

    #create interpolation function to CDF to find speeds
    inv_cdf = interp(cdf,vs,fill_value="extrapolate")

    rand_nums = np.random.random(n)
    speeds = inv_cdf(rand_nums)
    
    # spherical polar coords - generate random angle for velocity vector, uniformly distributed over the surface of a sphere
    phi = np.random.uniform(0,2*np.pi,n)
    alpha = np.arccos(np.random.uniform(-1,1,n))-np.pi/2
    
    # convert to cartesian units
    vx = speeds * np.sin(alpha) * np.cos(phi) 
    vy = speeds * np.sin(alpha) * np.sin(phi)
    vz = speeds * np.cos(alpha)
    
    return speeds, vx, vy, vz

#Vacuum system geometry

vac_diam=0.1; #diameter of vacuum system in general [m]
vac_len=0.5; #[m]

#Define atom class
class atom:
    def __init__(self,x,y,z,vx,vy,vz):
        """ This method is executed every time we create a new instance"""
        
        #Define species
        self.m=m_er;
        self.k=k_er;
        self.gamma=linewidth_er;
        self.Isat=Isat_er;

        #Define position/velocity
        self.x = x;
        self.y = y;
        self.z = z;
        self.vx = vx;
        self.vy = vy;
        self.vz = vz;

        #Define flags to determine how the simulation needs to continue
        #Ends if we hit a cold wall
        self.hotwallflag=0;
        self.coldwallflag=0;


def propagate_atom(atom,dt):

    #Update flags by checking if you're hitting a wall
    if (atom.z > cruc_len) and (atom.z < cruc_len+nozzle_len) and ((atom.x**2 + atom.y**2) >= 0.95*(hole_diam/2)**2): #0.95 is a fudge factor
      atom.hotwallflag=1;

    if (atom.z > 1.5*(cruc_len+nozzle_len)) or (atom.z < 0.9*cruc_len) or ((atom.x**2 + atom.y**2) >= 100*(hole_diam/2)**2): #100 is a graphing factor
      atom.coldwallflag=1;

    #Check hotwall flag
    if atom.hotwallflag==1:
      spd,vx_new,vy_new,vz_new=generate_velocities(1,atom.m,T_hl); #MB

      if ((atom.x*vx_new + atom.y*vy_new) > 0): #pos,vel dot product to ensure inward radial bounce
        vx_new = -vx_new;
        vy_new = -vy_new;

      atom.hotwallflag=0;

    else:
      vx_new=atom.vx
      vy_new=atom.vy
      vz_new=atom.vz

    #Update velocity
    atom.vx=vx_new; #3D
    #atom.vx=0; #2D
    atom.vy=vy_new;
    atom.vz=vz_new;

    #Update positions THIS USED TO COME FIRST
    atom.x=atom.x+atom.vx*dt;
    atom.y=atom.y+atom.vy*dt;
    atom.z=atom.z+atom.vz*dt;

    return atom

###############################################################################


def single_atom_simulation(dt=1e-9):

  #Generate atom, w/ random pos and escape vel
    x,y,z=generate_positions(1,hole_diam,cruc_len);
    spd,vx,vy,vz=generate_velocities(1,m_er,T_oven); #MB
    spd,vx,vy,vz=generate_velocities_ap(1,m_er,T_oven); #AP

    if vz<0:
      vx=-vx;
      vy=-vy;
      vz=-vz;

    atoms=atom(x,y,z,vx,vy,vz); #3D
    #atoms=atom(0,y,z,0,vy,vz); #2D

    #Trajectory
    xtraj=[];
    ytraj=[];
    ztraj=[];

    #Run dynamics
    counter=0;
    while (atoms.coldwallflag==0):

      xtraj.append(atoms.x);
      ytraj.append(atoms.y);
      ztraj.append(atoms.z);

      atoms=propagate_atom(atoms,dt);

      #have an iteration counter with a maximum for safe escape.
      counter=counter+1;
      if counter > (1e6): 
        atoms.coldwallflag=1; #call an atom dead after TIME
        #print('Timeout failure')

    return xtraj, ytraj, ztraj



num=24000;
div_angles=[]
for i in tqdm(range(num)):
  xtraj, ytraj, ztraj = single_atom_simulation();

  if ztraj[-1][0] > (cruc_len+nozzle_len):
    #Divergence angles
    vz_fin=ztraj[-1][0]-ztraj[-2][0]; #dt cancels out...
    vy_fin=ytraj[-1][0]-ytraj[-2][0];
    angle=np.arctan(vy_fin/vz_fin)*180/np.pi;
    div_angles.append(angle);



print(div_angles)

