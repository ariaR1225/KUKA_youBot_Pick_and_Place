import modern_robotics as mr
import numpy as np 
import csv
import math
import utils.milestone_config as milestone_config
import matplotlib.pyplot as plt
from datetime import datetime
import os

            
def to_csv(filename, res):
    """result save to csv

    Args:
        filename (str): filename of the generated csv
        res (list): list of trajectories to be saved
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for r in res:
            writer.writerow(r)
            
def T_decompose(T):
    """decompose transformation matrix T
    Args:
        T (np.ndarray): transformation matrix
        Ttest = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])
    Returns:
        list: decomposed ndarray
        [1, 2, 3, 5, 6, 7, 9, 10, 11, 4, 8, 12]
    """
    return [T[0,0], T[0, 1], T[0, 2], 
            T[1, 0], T[1, 1], T[1, 2],
            T[2, 0], T[2, 1], T[2, 2],
            T[0, 3], T[1, 3], T[2, 3]]
            
def T_compose(ls):
    """inverse of T_decompose
    make list into T

    Args:
        ls (list): one row of generated trajectory
    """
    return np.array([[ls[0], ls[1], ls[2], ls[-4]],
                     [ls[3], ls[4], ls[5], ls[-3]],
                     [ls[6], ls[7], ls[8], ls[-2]],
                     [0, 0, 0, 1]])
    
def T_sb_q(q):
    """Calculate T_sb

    Args:
        q (list): config

    Returns:
        np.array: T_sb
    """
    return np.array([[np.cos(q[0]), -np.sin(q[0]), 0, q[1]],
                     [np.sin(q[0]), np.cos(q[0]), 0, q[2]],
                     [0, 0, 1, .0963],
                     [0, 0 , 0, 1]])
    
def cap(x, x_cap):
    """cap x at x_cap

    Args:
        x (int): value to cap
        x_cap (int): maximum/minimum

    Returns:
        float: capped value
    """
    if x > x_cap:
        return float(x_cap)
    if x < -x_cap:
        return float(-x_cap)
    return float(x)

def testJointLimits(theta, rom):
    """
    Cap joint angles to stay within their range of motion limits

    Args:
        theta (list): joint ang
        rom (list): range of motion 

    Returns:
        list: capped joint ang

    Example:
        >>> theta = [1.5, -2.0, 0.5]
        >>> rom = [1.0, 1.0, 1.0]
        >>> collision_avoid(theta, rom)
        [1.0, -1.0, 0.5]
    """
    capped_theta = []
    for i in range(len(theta)):
        th = theta[i]
        if th > rom[i]:
            th = rom[i]
        if th < -rom[i]:
            th = -rom[i]
        capped_theta.append(th)  
    return capped_theta

def NextState(config, ctrl, dt, speed_lim, rom, _use_collision_avoid):
    """
    calculate the next state given current configuration and control input

    Args:
        config (numpy.ndarray): current robot config
        ctrl (list): ontrol input
        speed_lim (list): wheel max speed
        rom (list): joint rom
        _use_collision_avoid (bool): whether to apply joint angle limits

    Returns:
        numpy.ndarray: next state 
    """
    for i in range(len(speed_lim)):
        ctrl[i] = cap(ctrl[i], speed_lim[i])
    
    wheel_vel = ctrl[:4]
    Vb = milestone_config.F @ wheel_vel * dt # (3, 4) * (4, 1)   
    wbz = Vb[0]
    if math.isclose(wbz, 0, abs_tol=.0001): 
        dphib = 0
        dxb = Vb[1]
        dyb = Vb[2]
    else:
        dphib = Vb[0]
        dxb = (Vb[1]*np.sin(Vb[0]) + Vb[2]*(np.cos(Vb[0]) - 1))/Vb[0]
        dyb = (Vb[2]*np.sin(Vb[0]) + Vb[1]*(np.cos(Vb[0]) - 1))/Vb[0]
    dqb = np.array([dphib, dxb, dyb])
    dq = np.array([[1, 0, 0],
                [0, np.cos(config[0]), -np.sin(config[0])],
                [0, np.sin(config[0]), np.cos(config[0])]]) @ dqb

    # update 
    if _use_collision_avoid:
        new_joint = testJointLimits(config[3:8] + ctrl[4:]*dt, rom)
    else:
        new_joint = config[3:8] + ctrl[4:]*dt
        
    new_wheel = config[8:12]+ ctrl[:4]*dt
    new_chassis = config[:3] + dq
    
    return np.array([*new_chassis, *new_joint, *new_wheel, 1])

def TrajectoryGenerator(Tse_i, Tsc_i, Tsc_f, Tce_grasp, Tce_standoff, k = 1): 
    """Generates trajectory for the gripper to pick and place the cube 

    Args:
        Tse_i (np.ndarray): initial configuration of the end-effector in the reference trajectory
        Tsc_i (np.ndarray): cube's initial configuration
        Tsc_f (np.ndarray): cube's desired final configuration
        Tce_grasp (np.ndarray): end-effector's configuration relative to the cube when it is grasping the cube
        Tce_standoff (np.ndarray): end-effector's standoff configuration above the cube, before and after grasping, relative to the cube
        k (int): number of trajectory reference configurations per 0.01 seconds (defaults to 10)

    Returns:
        list: a list of trajectories 
            [r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper_state]
    """
    # trajectory set points
    Tse_standoff_grasp = Tsc_i @ Tce_standoff
    Tse_grasp_grasp = Tsc_i @ Tce_grasp
    Tse_standoff_goal = Tsc_f @ Tce_standoff
    Tse_grasp_goal = Tsc_f @ Tce_grasp
    steps = k/.01
    # Xstart, X_end, Tf, N, gripper
    traj_param = {"traj 1": [Tse_i, Tse_standoff_grasp, 8, 8*steps, 0], # traj 1: init -> standoff 2000ms
                  "traj 2": [Tse_standoff_grasp, Tse_grasp_grasp, 5, 5*steps, 0], # traj 2: standoff -> grasp 1000ms
                  "traj 3": [Tse_grasp_grasp, Tse_grasp_grasp, .65, .65*steps, 1], # traj 3: close gripper 650ms
                  "traj 4": [Tse_grasp_grasp, Tse_standoff_grasp, 2, 2*steps, 1], # traj 4: grasp -> standoff 1000ms
                  "traj 5": [Tse_standoff_grasp, Tse_standoff_goal, 10, 10*steps, 1], # traj 5: grasp standoff -> goal standoff 2000ms
                  "traj 6": [Tse_standoff_goal, Tse_grasp_goal, 5, 5*steps, 1], # traj 6: standoff -> goal 1000ms
                  "traj 7": [Tse_grasp_goal, Tse_grasp_goal, .65, .65*steps, 0], # traj 7: open gripper 650ms
                  "traj 8": [Tse_grasp_goal, Tse_standoff_goal, 1, 1*steps, 0]} # traj 8: goal -> standoff 1000ms
    time_scaling = 3 # cubic
    csv_res = []
    for i in range (1, 9):
        T_out = mr.CartesianTrajectory(*traj_param[f"traj {i}"][:-1], method=time_scaling)
        for t in T_out:
            traj_ls = T_decompose(t) # r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz
            traj_ls.append(traj_param[f"traj {i}"][-1]) # gripper_state
            csv_res.append(traj_ls)
        print(f"traj {i} done")
    print("Trajectory generated!")
    return csv_res

def FeedbackControl(X, Xd, Xd_next, Kp, Ki, dt):
    """ Feedforward + Feedback

    Args:
        X (np.array): actual X
        Xd (np.array): desired X
        Xd_next (np.array): next desired X
        Kp (np.array): proportional gain
        Ki (np.array): integral gain
        dt (float): timestep

    Returns:
        tuple: twist, error
    """
    Vd = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Xd)@Xd_next)/dt)
    Adj = mr.Adjoint(mr.TransInv(X) @ Xd)
    X_err = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(X) @ Xd))
    X_err_int = X_err + X_err*dt
    V = (Adj @ Vd) + (Kp @ X_err) + (Ki @ X_err_int)
    return V, X_err

def plot_err(err_int_ls, _type):
    """Error plot

    Args:
        err_int_ls (list): error list
        _type (str): task name
    """
    
    def tick_format(x, p):
        return f'{int(x*10)}'
    
    data_array = np.array(err_int_ls)
    x_points = range(len(err_int_ls))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    label1 = ["roll", "pitch", "yaw"]
    for i in range(3):
        ax1.plot(x_points, data_array[:, i], label=label1[i], linewidth=1)

    ax1.set_title("Angular Error (rad) v.s. Time(ms)")
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Angular Error (rad)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
    
    label2 = ["x", "y", "z"]
    for i in range(3, 6):
        ax2.plot(x_points, data_array[:, i], label=label2[i - 3], linewidth=1)

    ax2.set_title("Linear Error (m) v.s. Time(ms)")
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel("Linear Error (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
    
    plt.tight_layout()
    plt.savefig(f"result/{_type}/{_type}_error.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def write_readme(_type, task_config, Tsc_i, Tsc_f):
    """
    write README.txt
    
    Args:
        _type (str): task type
        task_config (dict): dictionary containing controller params
        Tsc_i (numpy.ndarray): initial cube config
        Tsc_f (numpy.ndarray): final cube config
    """
    with open(f"result/{_type}/{_type}_README.txt", 'w') as f:
        f.write("KUKA Youbot Controller\n")
        f.write("=========================================\n\n")
        
        f.write("Controller Parameters:\n")
        f.write("--------------------\n")
        f.write(f"Configuration Type: {_type}\n")
        f.write(f"Angular Feedback Gains:\n")
        f.write(f"  Kp: {task_config[_type]['kp_ang']}\n")
        f.write(f"  Ki: {task_config[_type]['ki_ang']}\n")
        f.write(f"Linear Feedback Gains:\n")
        f.write(f"  Kp: {task_config[_type]['kp_lin']}\n")
        f.write(f"  Ki: {task_config[_type]['ki_lin']}\n\n")
        
        f.write("Controller Settings:\n")
        f.write("-------------------\n")
        f.write(f"Controller Type: {task_config[_type]['ctrl_type']}\n")
        f.write(f"Initial Robot Configuration: {task_config[_type]['X_sim'][:3]}\n")
        f.write(f"Singularity Detection: {task_config[_type]['_use_singularity']}\n")
        f.write(f"Collision Avoidance: {task_config[_type]['_use_collision_avoid']}\n\n")

        f.write("Cube Configurations:\n")
        f.write("-------------------\n")
        f.write("Initial Cube Configuration (Tsc_i):\n")
        f.write(f"{np.array2string(Tsc_i, precision=4, separator=', ')}\n\n")
        f.write("Final Cube Configuration (Tsc_f):\n")
        f.write(f"{np.array2string(Tsc_f, precision=4, separator=', ')}\n")

def write_log(_type, info, first_write=False):
    """
    write log
    
    Args:
        _type (str): task type
        info: info to write in log
    """
    os.makedirs(f"result/{_type}", exist_ok=True)
    
    mode = 'w' if first_write else 'a'
    
    with open(f"result/{_type}/{_type}_log.txt", mode) as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {info}\n")
        f.write("---------------------------------------------\n")