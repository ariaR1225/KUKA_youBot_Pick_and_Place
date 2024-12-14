# lib
import modern_robotics as mr
import numpy as np 
import csv

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
    traj_param = {"traj 1": [Tse_i, Tse_standoff_grasp, 2, 2*steps, 0], # traj 1: init -> standoff 2000ms
                  "traj 2": [Tse_standoff_grasp, Tse_grasp_grasp, 1, 1*steps, 0], # traj 2: standoff -> grasp 1000ms
                  "traj 3": [Tse_grasp_grasp, Tse_grasp_grasp, .65, .65*steps, 1], # traj 3: close gripper 650ms
                  "traj 4": [Tse_grasp_grasp, Tse_standoff_grasp, 1, 1*steps, 1], # traj 4: grasp -> standoff 1000ms
                  "traj 5": [Tse_standoff_grasp, Tse_standoff_goal, 2, 2*steps, 1], # traj 5: grasp standoff -> goal standoff 2000ms
                  "traj 6": [Tse_standoff_goal, Tse_grasp_goal, 1, 1*steps, 1], # traj 6: standoff -> goal 1000ms
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
    return csv_res

if __name__ == "__main__":
    # init 
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
    thetas_standoff = np.pi/2 # grasp at 45 deg
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

    res = TrajectoryGenerator(Tse_i, Tsc_i, Tsc_f, Tce_grasp, Tce_standoff)
    to_csv("milestone2.csv", res)
    print("csv generated")