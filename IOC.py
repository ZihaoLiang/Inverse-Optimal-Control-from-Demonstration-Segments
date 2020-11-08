# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:00:50 2020

@author: Danieleung
"""

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from casadi import *


class IOC():
    
    def _init_(self, project_name = 'IOC'):
        self.prject_name = project_name

    def getdPhi(self, x_his, u_his, phi):
        
        X = MX.sym('X',len(x_his[0]),1)
        n = len(x_his[0])   
        U = MX.sym('U',len(u_his[0]),1)
                        
        PHI_sym = phi(X, U)
        
        self.dphidx_his = []
        self.dphidu_his = []
        
        for i in range(len(u_his)):
            
            dphidxi = []
            dphidui = []
            
            for j in range(len(PHI_sym)):
                dphidx = gradient(PHI_sym[j],X)
                dphidu = gradient(PHI_sym[j],U)           
        
                dphidx_num = Function('dphidx_num',[X,U],[dphidx])
                dphidu_num = Function('dphidu_num',[X,U],[dphidu]) 
    
                dphidxk = dphidx_num(x_his[i],u_his[i]).full()
                dphiduk = dphidu_num(x_his[i],u_his[i]).full()
                
                dphidxi.append(dphidxk.T)
                dphidui.append(dphiduk.T)
            
            dphidxi = np.concatenate(dphidxi,axis = 0)
            dphidui = np.concatenate(dphidui,axis = 0)
            
            self.dphidx_his.append(dphidxi)
            self.dphidu_his.append(dphidui)   

    def getdf(self, x_his, u_his, dyn):
           
        X = MX.sym('X',len(x_his[0]),1)
        U = MX.sym('U',len(u_his[0]),1)
            
        f = dyn(X, U)
        
        self.dfdx_his = []
        self.dfdu_his = []
        
        dfdx = jacobian(f,X)
        dfdu = jacobian(f,U)
                
        for i in range(len(u_his)):
                
            dfdx_num = Function('dfdx_num',[X,U],[dfdx],['X','U'],['dfdx'])
            dfdu_num = Function('dfdu_num',[X,U],[dfdu],['X','U'],['dfdu']) 
    
            dfdxk = dfdx_num(X = x_his[i],U = u_his[i])['dfdx'].full()
            dfduk = dfdu_num(X = x_his[i],U = u_his[i])['dfdu'].full()
                
            self.dfdx_his.append(dfdxk)
            self.dfdu_his.append(dfduk)
    
    def IOC_main(self, dfdx_his, dfdu_his, dphidx_his, dphidu_his):
        
        # T_seg = [[50,52],[10,30]]
        # T_seg = [[1,3],[10,13],[70,73],[80,83]]
        T_seg = [[1,30]]
        # T_seg = [[1,4],[10,13]]
        # T_seg = [[50,55],[90,92]]
        # T_seg = [[9,16],[26,39]]#,[21,41]]#,[20,40]]
            
        for seg in T_seg:
            T_start = seg[0]
            T_end = seg[1]
            
            dfui = self.dfdu_his[T_start]
            dfxi = self.dfdx_his[T_start + 1]
            dphixi = self.dphidx_his[T_start + 1]
            dphiui = self.dphidu_his[T_start]
            
            H1 = dfui.T@dphixi.T + dphiui.T
            H2 = dfui.T@dfxi.T
    
            for i in range(T_start + 1,T_end):
                dfui = self.dfdu_his[i]
                dfxi = self.dfdx_his[i + 1]
                dphixi = self.dphidx_his[i + 1]
                dphiui = self.dphidu_his[i]
                
                H1 = vertcat(H1 + H2@dphixi.T, dfui.T@dphixi.T + dphiui.T)
                H2 = vertcat(H2@dfxi.T, dfui.T@dfxi.T)
                         
            if seg == T_seg[0]:
                R = H1 - H2@inv(H2.T@H2)@H2.T@H1
                W = R.T@R
            else:
                R = H1 - H2@inv(H2.T@H2)@H2.T@H1  
                W += R.T@R
                        
        e1 = np.zeros((1, W.shape[0]), dtype=float)[0].reshape(W.shape[0],1)
        e1[0] = 1
        
        print(inv(W))
        self.omega = np.array(inv(W)@e1/(e1.T@inv(W)@e1)).T[0]
        