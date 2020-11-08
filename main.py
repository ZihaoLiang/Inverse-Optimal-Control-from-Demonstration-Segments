# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:15:43 2020

@author: Danieleung
"""

import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from dynamics_env import LTI, Pend, RobotArm, UAV, toQuaternion
from OptimalControl import OC
from IOC import IOC
    
dyn = 'Pendulum' #choose model dynamics: 'LTI', 'Pendulum', 'RobotArm', 'UAV'

############################### pendulum and LTI ###############################
if dyn == 'LTI' or dyn == 'Pendulum':
    init = [np.pi/4, np.pi/2]
    target = [0, 0]  

############################### Robot arm ###############################
elif dyn == 'RobotArm': 
    init = [np.pi/4, np.pi/2, 0, 0]
    target = [0, 0, 0, 0]  

############################### UAV ###############################    
elif dyn == 'UAV': 
    ini_p_I = [-8, -6, 9.]
    ini_v_I = [0.0, 0.0, 0.0]
    ini_q = toQuaternion(0, [1, -1, 1])
    ini_w = [0.0, 0.0, 0.0]
    init = ini_p_I + ini_v_I + ini_q + ini_w
    
    goal_p_I = [0, 0, 0]
    goal_v_I = [0, 0, 0]
    goal_q = toQuaternion(0, [0, 0, 1])
    goal_w_B = [0, 0, 0]
    target = goal_p_I + goal_v_I + goal_q + goal_w_B
   
############################### Perform optimal control ###############################    
T = 5 # number of control intervals
dt = 0.1
nT = int(10/0.1)

OCsys = OC()
OCsys.getTraj(dyn, init, target, T)
x_his = OCsys.x_his
u_his = OCsys.u_his

############################### IOC ###############################    
IOCsys = IOC()
IOCsys.getdPhi(OCsys.x_his, OCsys.u_his, OCsys.PHI)
IOCsys.getdf(OCsys.x_his, OCsys.u_his, OCsys.DYN)
IOCsys.IOC_main(IOCsys.dfdx_his, IOCsys.dfdu_his, IOCsys.dphidx_his, IOCsys.dphidu_his)

omega = IOCsys.omega / min(IOCsys.omega) #normalize omega
print('Omega =',omega)

############################### Pendulum and LTI trajectory ###############################
if dyn == 'LTI' or dyn == 'Pendulum':

    x_his = np.concatenate(x_his,axis = 1)
    u_his = np.concatenate(u_his,axis = 1)
    
    fig, axs = plt.subplots(2)

    axs[0].plot(x_his[0,:], label = 'x1')
    axs[0].plot(x_his[1,:], label = 'x2')
    axs[0].legend()
    axs[0].grid()
    axs[0].set(ylabel = 'x')

    axs[1].plot(u_his[0,:], label = 'u')
    axs[1].set(ylabel = 'u')
    axs[1].legend()
    axs[1].grid()
    
    plt.xlabel('t')
    plt.show()
    
############################### Robot Arm trajectory ###############################
elif dyn == 'RobotArm': 
  
    x_his = np.concatenate(x_his,axis = 1)
    u_his = np.concatenate(u_his,axis = 1)
    
    fig, axs = plt.subplots(2)

    axs[0].plot(x_his[0,:], label = 'q1')
    axs[0].plot(x_his[1,:], label = 'q2')
    axs[0].plot(x_his[2,:], label = 'dq1')
    axs[0].plot(x_his[3,:], label = 'dq2')
    axs[0].legend()
    axs[0].grid()
    axs[0].set(ylabel = 'x')

    axs[1].plot(u_his[0,:], label = 'u1')
    axs[1].plot(u_his[1,:], label = 'u2')
    axs[1].set(ylabel = 'u')
    axs[1].legend()
    axs[1].grid()
    
    plt.xlabel('t')
    plt.show()

############################### UAV trajectory ###############################
elif dyn == 'UAV': 

    x_his = np.concatenate(x_his,axis = 1)
    u_his = np.concatenate(u_his,axis = 1)
    
    fig, axs = plt.subplots(4)

    axs[0].plot(x_his[0,:], label = 'p1')
    axs[0].plot(x_his[1,:], label = 'p2')
    axs[0].plot(x_his[2,:], label = 'p3')
    axs[0].legend()
    axs[0].grid()
    axs[0].set(ylabel = 'p')
    
    axs[1].plot(x_his[3,:], label = 'v1')
    axs[1].plot(x_his[4,:], label = 'v2')
    axs[1].plot(x_his[5,:], label = 'v3')
    axs[1].legend()
    axs[1].grid()
    axs[1].set(ylabel = 'v')
    
    axs[2].plot(x_his[6,:], label = 'q1')
    axs[2].plot(x_his[7,:], label = 'q2')
    axs[2].plot(x_his[8,:], label = 'q3')
    axs[2].plot(x_his[9,:], label = 'q4')
    axs[2].legend()
    axs[2].grid()
    axs[2].set(ylabel = 'q')
    
    axs[3].plot(x_his[10,:], label = 'w1')
    axs[3].plot(x_his[11,:], label = 'w2')
    axs[3].plot(x_his[12,:], label = 'w3')
    axs[3].legend()
    axs[3].grid()
    axs[3].set(ylabel = 'w')
    
    plt.xlabel('t')
    plt.show()

    plt.plot(u_his[0,:], label = 'T1')
    plt.plot(u_his[1,:], label = 'T2')
    plt.plot(u_his[2,:], label = 'T3')
    plt.plot(u_his[3,:], label = 'T4')
    plt.legend()
    plt.grid()
    
    plt.xlabel('t')
    plt.show()