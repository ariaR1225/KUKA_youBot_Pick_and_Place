import numpy as np
import modern_robotics as mr 


task_config = {"overshoot": {"X_sim": np.array([-np.pi/4, 1., .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             "kp_ang": 1,
                             "ki_ang": 1,
                             "kp_lin": 1,
                             "ki_lin": 1,
                             "_use_singularity": True,
                             "_use_collision_avoid": False,
                             "ctrl_type": "Feedforward + angular PI + linear PI"},
               "best": {"X_sim": np.array([-np.pi/4, 1., .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             "kp_ang": .85,
                             "ki_ang": .05,
                             "kp_lin": .9,
                             "ki_lin": .0,
                             "_use_singularity": True,
                             "_use_collision_avoid": False,
                             "ctrl_type": "Feedforward + angular PI + linear P"},
               "newTask": {"X_sim": np.array([np.pi/4, -1., -.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             "kp_ang": .4,
                             "ki_ang": .01,
                             "kp_lin": .9,
                             "ki_lin": .0,
                             "_use_singularity": True,
                             "_use_collision_avoid": False,
                             "ctrl_type": "Feedforward + angular PI + linear P"},
               "improved": {"X_sim": np.array([-np.pi/4, 1., .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             "kp_ang": 1.35,
                             "ki_ang": .13,
                             "kp_lin": .18,
                             "ki_lin": .15,
                             "_use_singularity": True,
                             "_use_collision_avoid": True,
                             "ctrl_type": "Feedforward + angular PI + linear PI"}}

# fixed offset from the chassis frame {b} to the base frame of the arm {0}
Tb0 = np.array([[1, 0, 0, .1662],
                [0, 1, 0, 0],
                [0, 0, 1, .0026],
                [0, 0, 0, 1]])
# arm is at its home configuration (all joint angles zero, as shown in the figure), the end-effector frame {e} relative to the arm base frame {0
M0e = np.array([[1, 0, 0, .033],
                [0, 1, 0, 0],
                [0, 0, 1, .6546],
                [0, 0, 0, 1]])

# at home configuration, the screw axes B for the five joints are expressed in the end-effector frame {e} 
B1 = [0 ,0 , 1, 0, .033, 0]
B2 = [0, -1, 0, -.5076, 0, 0]
B3 = [0, -1, 0, -.3526, 0, 0]
B4 = [0, -1, 0, -.2176, 0, 0]
B5 = [0 ,0 , 1, 0, 0, 0]
Blist = np.array([B1, B2, B3, B4, B5]).T

# chassis
L = .47/2
W = .3/2
R = .0475
F = (R/4)*np.array([[-1/(L + W), 1/(L + W), 1/(L + W), -1/(L + W)],
                    [1, 1, 1, 1],
                    [-1, 1, -1, 1]])

F6 = np.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               F[0],
               F[1],
               F[2],
               [0 , 0, 0, 0]])

# milestone 2 pick and place param
Tse_i = np.array([[0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [-1, 0, 0, .5],
                    [0, 0, 0, 1]])

Tsc_i = np.array([[1, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 1, .025],
                [0, 0, 0, 1]])

Tsc_f = np.array([[0, 1, 0, 0],
                [-1, 0, 0, -1],
                [0, 0, 1, .025],
                [0, 0, 0, 1]])

# ee0
Rs_ee = np.array([[0, 0, 1],
                    [0, 1, 0],
                    [-1, 0, 0]])
# ee at standoff
thetas_standoff = np.pi/4 # grasp at 45 deg
Rec_standoff = np.array([[-np.cos(thetas_standoff), 0, -np.sin(thetas_standoff)],
                        [0, 1, 0],
                        [np.sin(thetas_standoff), 0, -np.cos(thetas_standoff)]])

## ee to standoff
Ree_stanoff = mr.RotInv(Rs_ee) @ Rec_standoff
Tce_standoff = Tce_grasp = np.array([[Ree_stanoff[0, 0], Ree_stanoff[0, 1], Ree_stanoff[0, 2], 0],
                                    [Ree_stanoff[1, 0], Ree_stanoff[1, 1], Ree_stanoff[1, 2], 0],
                                    [Ree_stanoff[2, 0], Ree_stanoff[2, 1], Ree_stanoff[2, 2], .25],
                                    [0, 0, 0, 1]])
Tce_grasp = np.array([[Ree_stanoff[0, 0], Ree_stanoff[0, 1], Ree_stanoff[0, 2], 0],
                        [Ree_stanoff[1, 0], Ree_stanoff[1, 1], Ree_stanoff[1, 2], 0],
                        [Ree_stanoff[2, 0], Ree_stanoff[2, 1], Ree_stanoff[2, 2], 0],
                        [0, 0, 0, 1]])

dt = .01
k = 1
speed_lim = 10
wheel_speed_lim = [speed_lim, speed_lim, speed_lim, speed_lim]
rom = [3.0, 1.6, 2.6, 1.9, 2.9] # reference: https://www.cyberbotics.com/doc/guide/youbot?version=cyberbotics:R2019a-rev1