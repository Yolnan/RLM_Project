import numpy as np
import gtsam


class map:
    def __init__(self):
        self.points = np.empty((0,3))
        self.Rs = np.empty((0,3))
        self.ts = np.empty((0,3))
        self.pose_ids = []
    

    def update_map(self, result):
        for ind in range(self.ts.shape[0]):
            x_n = gtsam.symbol('x', self.pose_ids[ind])
            R_new = result.atPose3(x_n).rotation().matrix()
            t_new = result.atPose3(x_n).translation()
            self.Rs[ind*3:ind*3+3,:] = R_new
            self.ts[ind,:] = np.expand_dims(t_new, axis=0)

        for ind in range(self.points.shape[0]):
            l_n = gtsam.symbol('l',ind)
            self.points[ind,:] = result.atPoint3(l_n)

    def add_landmarks(self, new_points):
        self.points = np.concatenate((self.points, new_points), axis = 0)


    def add_pose(self, new_R, new_t, pose_id ):
        self.Rs = np.concatenate((self.Rs, new_R), axis = 0)
        self.ts = np.concatenate((self.ts, new_t), axis = 0)
        self.pose_ids.append(pose_id)

        