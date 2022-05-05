import numpy as np
import matplotlib.pyplot as plt
import glob

# load traj estimation using essential matrix init estimate
traj_est_ess = np.load("local_map_ts_ess.npy")

# load traj estimation using RGB-D odometry init estimate
traj_est_rgbdo = np.load("local_map_ts_rgbdo.npy")

# load traj estimation using ground truth init estimate
traj_est_gt = np.load("local_map_ts_gt.npy")

# ORB SLAM trajectory
traj_orbslam = np.loadtxt("KeyFrameTrajectory_corner0.txt")[:,1:4]

# load ground truth
filenames = glob.glob("./absolute_pose/*.txt")
filenames.sort()
traj_gt = [np.loadtxt(file)[:, 3] for file in filenames]
traj_gt = np.vstack(traj_gt)


## plot results

# traj estimate using essential matrix init estimate
fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot3D(traj_est_ess[:,0], traj_est_ess[:,1], traj_est_ess[:,2], 'red')
ax.plot3D(traj_gt[:,0], traj_gt[:,1], traj_gt[:,2], 'blue')
ax.plot3D(traj_gt[0,0], traj_gt[0,1], traj_gt[0,2], 'k*')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
plt.legend(["estimated", "ground truth", "initial position"])
plt.show()

# traj estimate using RGB-D odometry init estimate
fig = plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot3D(traj_est_rgbdo[:,0], traj_est_rgbdo[:,1], traj_est_rgbdo[:,2], 'red')
ax.plot3D(traj_gt[:,0], traj_gt[:,1], traj_gt[:,2], 'blue')
ax.plot3D(traj_gt[0,0], traj_gt[0,1], traj_gt[0,2], 'k*')
plt.legend(["estimated", "ground truth", "initial position"])
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
plt.show()

# traj estimate using ground truth init estimate
fig = plt.figure(3)
ax = plt.axes(projection='3d')
ax.plot3D(traj_est_gt[:,0], traj_est_gt[:,1], traj_est_gt[:,2], 'r-')
ax.plot3D(traj_gt[:,0], traj_gt[:,1], traj_gt[:,2], 'b--')
ax.plot3D(traj_gt[0,0], traj_gt[0,1], traj_gt[0,2], 'k*')
plt.legend(["estimated", "ground truth", "initial position"])
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
plt.show()

fig = plt.figure(4)
traj_gt = traj_gt - traj_gt[0,:]    # shifted to place initial point at zero, ground truth doesn't start at 0,0,0
ax = plt.axes(projection='3d')
ax.plot3D(traj_orbslam[:,2], -traj_orbslam[:,0], -traj_orbslam[:,1], 'green')
ax.plot3D(traj_gt[:,0], traj_gt[:,1], traj_gt[:,2], 'b')
ax.plot3D(traj_gt[0,0], traj_gt[0,1], traj_gt[0,2], 'k*')
plt.legend(["ORB-SLAM", "ground truth", "initial position"])
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
plt.show()