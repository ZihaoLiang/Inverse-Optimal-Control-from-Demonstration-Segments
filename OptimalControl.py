# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:57:54 2020

@author: Danieleung
"""
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from dynamics_env import LTI, Pend, RobotArm, UAV, toQuaternion

class OC():
    def _init_(self, project_name = 'Optimal Control'):
        self.prject_name = project_name
        
    def getTraj(self, dyn, init, target, T):
        
        dt = 0.1
        nT = int(T/dt)
                
        #set environment
        if dyn == 'LTI':
            self.sys = LTI()
            self.w = [2, 1, 1]   #weight vector

        elif dyn == 'RobotArm':
            self.sys = RobotArm()
            self.w = [2, 1, 1, 1, 1]   #weight vector

        elif dyn == 'Pendulum':
            self.sys = Pend()
            self.w = [2, 1, 1]   #weight vector

        elif dyn == 'UAV':
            self.sys = UAV()
            self.w = [2, 1, 1, 1]   #weight vector

        else:
            print('Unknown dynamics')

        self.sys.Dyn(nT)
        self.X = self.sys.X
        self.U = self.sys.U
        opti = self.sys.opti
                
        for i in range(nT):
                
            #declare discrete model equations
            self.x_next = self.X[:,i] + dt*self.sys.f[i]
    
            opti.subject_to(self.X[:,i + 1] == self.x_next)
            
        self.sys.Cost(self.w, target, nT)
        self.Cost = self.sys.cost
        
        self.DYN = Function('dyn', [self.X, self.U], [self.x_next])
        self.PHI = Function('phi', [self.X, self.U], self.sys.phi)   
        self.OBS = Function('obs', [self.X], [self.sys.obs])
    
        opti.subject_to(self.X[:,0] == init)
        
        opti.minimize(self.Cost)
        
        p_opts = {"expand":True}
        s_opts = {"max_iter":10000,"print_level": 1}
        opti.solver('ipopt',p_opts,s_opts)
            
        sol = opti.solve()
        self.u_opt = sol.value(self.U)
        self.x_opt = sol.value(self.X)
        
        self.x_his = []
        for i in range(nT + 1):
            self.x_his.append(self.x_opt[:,i].reshape(self.X.shape[0],1))
        
        self.u_his = []
        if self.U.shape[0] == 1:                
            for i in range(nT):
                self.u_his.append(self.u_opt[i].reshape(self.U.shape[0],1))
        else:
            for i in range(nT):
                self.u_his.append(self.u_opt[:,i].reshape(self.U.shape[0],1))
        
            
