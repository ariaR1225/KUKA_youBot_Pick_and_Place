import modern_robotics as mr 
import numpy as np 
import utils.milestone_func as ms
from tqdm import tqdm
from utils.milestone_config import *

"""
To run this code: 
1. make sure you are at the main directory *-Ruan_Aria_Final_Proj-*
2. create and activate the environment
3. run this script

cd Ruan_Aria_Final_Proj
conda env create -f environment.yaml
conda activate MR_env
python best.py
"""

# init
TYPE = "best"
print("------------------------------------------------------------")
print(f"Running {TYPE}...")
print("------------------------------------------------------------")
ms.write_log(TYPE, f"Running {TYPE}...", first_write=True)

X_sim = task_config[TYPE]["X_sim"]
kp_ang = task_config[TYPE]["kp_ang"]
ki_ang = task_config[TYPE]["ki_ang"]
kp_lin = task_config[TYPE]["kp_lin"]
ki_lin = task_config[TYPE]["ki_lin"]
print(f"ang kp: {kp_ang}, ang ki: {ki_ang}\nlin kp: {kp_lin}, lin ki: {ki_lin}\nRobot init at {X_sim[:3]}")

_use_singularity = task_config[TYPE]["_use_singularity"]
_use_collision_avoid = task_config[TYPE]["_use_collision_avoid"]
print(f"Detect singularities? {_use_singularity}\nAvoid self-collision? {_use_collision_avoid}")
print("------------------------------------------------------------")

X_err_int = 0
err_ls = []
output_ls = []

Kp = np.diag([kp_ang, kp_ang, kp_ang,    # ang 
              kp_lin, kp_lin, kp_lin])    # lin 

Ki = np.diag([ki_ang, ki_ang, ki_ang,     # ang 
              ki_lin, ki_lin, ki_lin])     # lin

# traj gen
traj = ms.TrajectoryGenerator(Tse_i, Tsc_i, Tsc_f, Tce_grasp, Tce_standoff, k)
print("Tajectory generated.")
print(f"Initial reference: {traj[0][:3]}")
print("------------------------------------------------------------")
ms.write_log(TYPE, f"Tajectory generated.")

# controller
ms.write_log(TYPE, f"Processing trajectory...")
for i in tqdm(range(len(traj) - 1), desc="Processing trajectory"):
    T0e = mr.FKinBody(M0e, Blist, X_sim[3:8])
    Tsb = ms.T_sb_q(X_sim[:3])
    X = Tsb @ Tb0 @ T0e # Tse = Tsb @ Tb0 @ T0e
    Xd = ms.T_compose(traj[i])
    Xd_next = ms.T_compose(traj[i + 1])
    V, X_err = ms.FeedbackControl(X, Xd, Xd_next, Kp, Ki, dt)
    X_err_int += X_err*dt
    J_base = mr.Adjoint(mr.TransInv(T0e) @ mr.TransInv(Tb0)) @ F6
    J_arm = mr.JacobianBody(Blist, X_sim[3:8])
    Je = np.hstack((J_base, J_arm))
    
    if _use_singularity:
        Je_inv = np.linalg.pinv(Je, rcond=.01)
    else: 
        Je_inv = np.linalg.pinv(Je)
        
    ctrl = Je_inv @ V
    X_sim = ms.NextState(X_sim, ctrl, dt, wheel_speed_lim, rom, _use_collision_avoid)
    X_sim[-1] = traj[i][-1]
    
    if i%k == 0:
        err_ls.append(X_err)
        output_ls.append(X_sim)
        
# file output        
print("Writing to csv...")
ms.write_log(TYPE, "Writing to csv...")

ms.to_csv(f"result/{TYPE}/{TYPE}_result.csv", output_ls)
print("Result saved.")
ms.write_log(TYPE, "Result saved.")

ms.to_csv(f"result/{TYPE}/{TYPE}_error.csv", err_ls)
print("Error saved.")
ms.write_log(TYPE, "Error saved.")

ms.plot_err(err_ls, TYPE)
print("Plot saved.")
ms.write_log(TYPE, "Plot saved.")

ms.write_readme(TYPE, task_config, Tsc_i, Tsc_f)
print("README saved.")
ms.write_log(TYPE, "README saved.")

print(f"Directory: result/{TYPE}\n***DONE!***")
print("------------------------------------------------------------")
ms.write_log(TYPE, f"Directory: result/{TYPE}\n***DONE!***")
