# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:32:29 2020

@author: Danieleung
"""

import numpy as np
from casadi import *
import matplotlib.pyplot as plt
import math


############################### LTI system ###############################
class LTI:
    def _init_(self, project_name = 'LTI'):
        self.prject_name = project_name
        
    def Dyn(self, nT):
        
        #declare variables
        self.opti = casadi.Opti()
        self.X = self.opti.variable(2, nT + 1)
        self.U = self.opti.variable(1, nT) #input
        
        #continuous dynamics model
        self.f = []
        for i in range(nT):
            self.f.append(vertcat(self.X[1,i], self.U[:,i]))
            
        self.obs = vcat([]) #observables

    def Cost(self, w, target, nT):
                
        self.cost = 0
        self.x1 = self.X[0,:]
        self.x2 = self.X[1,:]

        for i in range(nT):

            # cost for x1
            self.cost_x1 = (self.x1[i] - target[0]) ** 2
            # cost for x2
            self.cost_x2 = (self.x2[i] - target[1]) ** 2
            # cost for u
            self.cost_u = dot(self.U[i], self.U[i])
            
            self.phi = np.array([self.cost_x1, self.cost_x2,self.cost_u])  #feature

            self.cost += w @ self.phi
            
############################### Pendulum ###############################
class Pend:
    def _init_(self, project_name = 'Pend'):
        self.prject_name = project_name
        
    def Dyn(self, nT):
        
        #declare variables
        self.opti = casadi.Opti()
        self.X = self.opti.variable(2, nT + 1)
        self.U = self.opti.variable(1, nT) #input
        
        #set global parameters
        g = 10
    
        #declare the pendulum parameters
        m, l, d, wth, wthdot, wu = 2, 1, 0.5, 3, 1, 0.5
            
        #continuous dynamics model
        self.f = []
        for i in range(nT):
            x1 = self.X[0,i]
            x2 = self.X[1,i]
            self.f.append(vertcat(x2, (-g/l*sin(x1) - d/(m*l*l)*x2 + 1/(m*l*l)*self.U[:,i])))
            
        self.obs = vcat([sin(x1), x1**2, x2**2, x1 + x2]) #observables

    def Cost(self, w, target, nT):
                
        self.cost = 0
        self.x1 = self.X[0,:]
        self.x2 = self.X[1,:]

        for i in range(nT):

            # cost for x1
            self.cost_x1 = (self.x1[i] - target[0]) ** 2
            # cost for x2
            self.cost_x2 = (self.x2[i] - target[1]) ** 2
            # cost for u
            self.cost_u = dot(self.U[i], self.U[i])
            
            self.phi = np.array([self.cost_x1, self.cost_x2,self.cost_u])  #feature

            self.cost += w @ self.phi
            
############################### RobotArm ###############################
class RobotArm:
    def _init_(self, project_name = 'RobotArm'):
        self.prject_name = project_name
        
    def Dyn(self, nT):
        
        #declare variables
        self.opti = casadi.Opti()
        self.X = self.opti.variable(4, nT + 1)
        self.U = self.opti.variable(2, nT) #input
        
        #set global parameters
        g = 0
        
        #declare the robot arm parameters
        m1, m2, l1, l2, lc1, lc2, I1, I2 = 1, 1, 1, 1, 0.5, 0.5, 1/12, 1/12
           
        #dynamics model
        a1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2) + I1 + I2
        a2 = m2 * l1 * lc2
        a3 = m2 * lc2**2 + I2
        b1 = m1 * lc1 + m2 * l1
        b2 = m2 * lc2
            
        #continuous dynamics model
        self.f = []
        for i in range(nT):
            q1 = self.X[0,i]
            q2 = self.X[1,i]
            dq1 = self.X[2,i]
            dq2 = self.X[3,i]
            
            M11 = a1 + 2 * a2 * cos(q2)
            M12 = a3 + a2 * cos(q2)
            M21 = M12
            M22 = a3
            invM11 = M22 / (M11 * M22 - M12 * M21)
            invM12 = -M12 / (M11 * M22 - M12 * M21)
            invM21 = -M21 / (M11 * M22 - M12 * M21)
            invM22 = M11 / (M11 * M22 - M12 * M21)
            C11 = -a2 * dq2 * sin(q2)
            C12 = -a2 * (dq1 + dq2) * sin(q2)
            C21 = a2 * dq1 * sin(q2)
            C22 = 0
            C = vertcat(horzcat(C11,C12),horzcat(C21,C22))
            
            G1 = b1 * g * cos(q1) + b2 * g * cos(q1 + q2)
            G2 = b2 * g * cos(q1 + q2)
            G = vertcat(G1,G2)
            ddq1 = horzcat(invM11,invM12)@(-C@vertcat(dq1,dq2) - G + self.U[:,i])
            ddq2 = horzcat(invM21,invM22)@(-C@vertcat(dq1,dq2) - G + self.U[:,i])
                        
            self.f.append(vertcat(dq1,dq2,ddq1,ddq2))
            
        self.obs = vcat([cos(q2), dq2 * sin(q2), (dq1 + dq2) * sin(q2), dq1 * sin(q2), cos(q1), cos(q1 + q2), dq1 + dq2]) #observables

    def Cost(self, w, target, nT):
                
        self.cost = 0       

        for i in range(nT):
            
            self.q1 = self.X[0,i]
            self.q2 = self.X[1,i]
            self.dq1 = self.X[2,i]
            self.dq2 = self.X[3,i]
           
            # state cost
            # self.goal_q_1 = np.array(target[0])
            self.cost_q_1 = dot(self.q1 - target[0], self.q1 - target[0])
            
            # goal_q_2 = np.array(target[1])
            self.cost_q_2 = dot(self.q2 - target[1], self.q2 - target[1])
            
            # goal_dq_1 = np.array(target[2])
            self.cost_dq_1 = dot(self.dq1 - target[2], self.dq1 - target[2])
            
            # goal_dq_2 = np.array(target[3])
            self.cost_dq_2 = dot(self.dq2 - target[3], self.dq2 - target[3])
            
            # input cost
            self.cost_u = dot(self.U[:,i], self.U[:,i])
            
            self.phi = np.array([self.cost_q_1, self.cost_q_2, self.cost_dq_1, self.cost_dq_2,self.cost_u])  #feature
            
            self.cost += w @ self.phi

############################### UAV ###############################
class UAV:
    def _init_(self, project_name = 'UAV'):
        self.prject_name = project_name
    
    def Dyn(self, nT):
        
        #declare variables
        self.opti = casadi.Opti()
        self.X = self.opti.variable(13, nT + 1)
        self.U = self.opti.variable(4, nT) #input
        
        #define known values
        J_B = diag(vertcat(1, 1, 5))
        g = 10
        g_I = vertcat(0, 0, -g)
        l = 0.4
        c = 0.01
        m = 1
            
        #continuous dynamics model
        self.f = []
        for i in range(nT):
            p_I = self.X[0:3,i] #displacement   
            v_I = self.X[3:6,i] #velocity
            q = self.X[6:10,i] #quarternion
            w_B = self.X[10:13,i] #angular velocity
            
            T_B = self.U[:,i]
            
            # total thrust in body frame
            thrust = T_B[0] + T_B[1] + T_B[2] + T_B[3]
            thrust_B = vertcat(0, 0, thrust)
            
            # total moment M in body frame
            Mx = -T_B[1] * l / 2 + T_B[3] * l / 2
            My = -T_B[0] * l / 2 + T_B[2] * l / 2
            Mz = (T_B[0] - T_B[1] + T_B[2] - T_B[3]) * c
            M_B = vertcat(Mx, My, Mz)
        
            # cosine directional matrix
            C_B_I = self.dir_cosine(q)  # inertial to body
            C_I_B = transpose(C_B_I)  # body to inertial
        
            # Newton's law
            dp_I = v_I
            dv_I = 1 / m * C_I_B@thrust_B + g_I
            
            # Euler's law
            dq = 1 / 2 * self.Omega(w_B)@q
            dw = inv(J_B)@(M_B - skew(w_B)@J_B@w_B)
                                
            self.f.append(vertcat(dp_I, dv_I, dq, dw))
            
        self.obs = vcat([p_I[0]**2, p_I[1]**2, p_I[2]**2, v_I[0]**2, v_I[1]**2, v_I[2]**2]) #observables
        
    def Cost(self, w, target, nT):
                
        self.cost = 0       

        for i in range(nT):
            
            self.p_I = self.X[0:3,i] #displacement   
            self.v_I = self.X[3:6,i] #velocity
            self.q = self.X[6:10,i] #quarternion
            self.w_B = self.X[10:13,i] #angular velocity
            self.T_B = self.U[:,i]
      
            # goal position in the world frame
            self.goal_p_I = np.array(target[0:3])
            self.cost_p_I = dot(self.p_I - self.goal_p_I, self.p_I - self.goal_p_I)
            
            # goal velocity
            self.goal_v_I = np.array(target[3:6])
            self.cost_v_I = dot(self.v_I - self.goal_v_I, self.v_I - self.goal_v_I)
            
            # final attitude error
            self.goal_q = target[6:10]
            self.goal_R_B_I = self.dir_cosine(self.goal_q)
            self.R_B_I = self.dir_cosine(self.q)
            self.cost_q = trace(np.identity(3) - transpose(self.goal_R_B_I)@self.R_B_I)
            
            # auglar velocity cost
            self.goal_w_B = np.array(target[10:13])
            self.cost_w_B = dot(self.w_B - self.goal_w_B, self.w_B - self.goal_w_B)
    
            # the thrust cost
            self.cost_thrust = dot(self.T_B, self.T_B)
            
            self.phi = np.array([self.cost_p_I, self.cost_v_I, self.cost_q, self.cost_thrust])   #feature
                        
            self.cost += w @ self.phi            
    
    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I
    
    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross
    
    def Omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg
    
    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0])

############################### Other functions ###############################

# converter to quaternion from (angle, direction)
def toQuaternion(angle, dir):
    if type(dir) == list:
        dir = numpy.array(dir)
    dir = dir / numpy.linalg.norm(dir)
    quat = numpy.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat.tolist()