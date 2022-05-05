import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import gtsam
from frame_to_frame import *
from map import *


def convert2stereoPt(uL, v, bf, depth):
    uR = uL - bf/depth 
    return gtsam.StereoPoint2(uL, uR, v)

def Rt2T(R, t):
    return np.concatenate((R, np.expand_dims(t, axis = 1)), axis = 1 )


def find_ORB_kps(image):
    orb = cv2.ORB_create()
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)
    return kp, des


if __name__ == '__main__':
    # load and sort rgb and depth images
    path_data_folder = "/home/yolnanc/void-dataset/void_release/void_150/data/corner0/"
    rgb_paths = glob.glob(path_data_folder + "/image/*.png")
    rgb_paths.sort()
    images = [cv2.imread(path, 0) for path in rgb_paths]   # read as greyscale

    depth_paths = glob.glob(path_data_folder + "sparse_depth/*.png")
    depth_paths.sort()
    depth_imgs = [cv2.imread(dpath,-1) for dpath in depth_paths]

    gt_pose_paths = glob.glob(path_data_folder + "absolute_pose/*.txt")
    gt_pose_paths.sort()
    gt_poses = [np.loadtxt(pose) for pose in gt_pose_paths]
    gt_pose_init = gt_poses[0]

    # camera properties
    baseline = 50  # width between stereo IR imagers, mm
    focal_length = 1.93    # focal length, mm
    K = np.array([[5.533317252305629381e+02, 0.0, 3.286542425444330320e+02],[0.0, 5.507826069073726103e+02, 2.520388973352311268e+02],[0.0, 0.0, 1.000000000000000000e+00]])    # intrinsics
    f = np.average([K[0,0], K[1,1]])    # average focal length x,y in pixels
    pixel_per_mm_scale =  f/focal_length
    b = baseline*pixel_per_mm_scale
    bf = b*f

    print("number of images: " + str(len(images)))
    # process 1st frame keypoints
    local_map = map()   
    kp_prev, des_prev = find_ORB_kps(images[0])
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # local_map.add_pose(np.eye(3), np.zeros((1,3)), 0)  # initialize 1st pose as identity   
    local_map.add_pose(gt_pose_init[:,0:3], np.expand_dims(gt_pose_init[:,3], axis=0), 0)         # initialize 1st pose using ground truth       
    kp_prev_land_id  = -np.ones((len(kp_prev),),int)          # landmark ID for initial frame
    
    pose_init_R = []
    pose_init_t = []
    pose_init_R.append(local_map.Rs[0:3,:])
    pose_init_t.append(local_map.ts[0,:])

    # camera intrinsics and measurement noise model
    # format: fx fy skew cx cy baseline
    K_stereo = gtsam.Cal3_S2Stereo(5.533317252305629381e+02, 5.507826069073726103e+02, 0, 3.286542425444330320e+02, 2.520388973352311268e+02, baseline*1e-3)
    stereo_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))

    # Create graph container 
    graph = gtsam.NonlinearFactorGraph()

    # add a constraint on the starting pose
    first_pose = gtsam.Pose3(gtsam.Rot3(local_map.Rs[0:3,0:3]), local_map.ts[0,:])
    x0 = gtsam.symbol('x',0) 
    graph.add(gtsam.NonlinearEqualityPose3(x0, first_pose))

    # add initial pose measurements to graph
    for ind in range(len(kp_prev)):
        # add initial landmarks to graph
        [uL, v] = kp_prev[ind].pt
        Z = depth_imgs[0][int(v),int(uL)]
        if Z != 0:
            kp_prev_land_id[ind] = local_map.points.shape[0]
            l = gtsam.symbol('l', kp_prev_land_id[ind])
            kp_stereo = convert2stereoPt(uL, v, bf, Z*1000*pixel_per_mm_scale)    # is this a 3D point, x,y,z?
            graph.add(gtsam.GenericStereoFactor3D(kp_stereo, stereo_model, x0, l, K_stereo))
            # add initial landmarks to map
            Stereo_Camera = gtsam.StereoCamera(first_pose, K_stereo)
            local_map.add_landmarks(np.expand_dims(Stereo_Camera.backproject(kp_stereo), axis=0))
        else:
            kp_prev_land_id[ind] = -2
    

    ## iterate through frames ##
    for i in range(1, len(images)):
        # compute orb features of current frame
        kp_curr, des_curr = find_ORB_kps(images[i])
        # compute initial pose of current frame
        # F = compute_F(images[i-1],images[i])
        # E = essentialMatrix(F)
        # R_init, t_init = computeRt(E)
        # R_init, t_init = computeRt(images[i-1], images[i])
        R_init, t_init = computeRt(images[i-1], images[i], depth_imgs[i-1], depth_imgs[i], K)
        pose_init_R.append(R_init)
        pose_init_t.append(t_init)

        # ground truth poses as 'estimates'
        R_init = gt_poses[i][:,0:3]
        t_init = gt_poses[i][:,3]   

        has_valid_measurement = False   # boolean flag to indicate whether current pose has valid correspondences

        # find matching features between current frame and previous
        matches = matcher.match(des_prev , des_curr)
        kp_curr_land_id = -np.ones((len(kp_curr),), int)  # landmark ID for current frame
        x_curr = gtsam.symbol('x', i) 

        # iterate through matched keypoints
        for match in matches:
            [uL_prev, v_prev] = kp_prev[match.queryIdx].pt
            [uL_curr, v_curr] = kp_curr[match.trainIdx].pt
            Z = depth_imgs[i][int(v_curr), int(uL_curr)]
            if Z != 0 and kp_prev_land_id[match.queryIdx] >= 0:
                # add previously seen landmark measurements to graph
                l = gtsam.symbol('l', kp_prev_land_id[match.queryIdx])
                kp_curr_land_id[match.trainIdx] = kp_prev_land_id[match.queryIdx]
                kp_stereo = convert2stereoPt(uL_curr, v_curr, bf, Z*1000*pixel_per_mm_scale)
                graph.add(gtsam.GenericStereoFactor3D(kp_stereo, stereo_model, x_curr,l, K_stereo))
                has_valid_measurement = True
            else:
                kp_curr_land_id[match.trainIdx] = -2

        # find indices of new keypoints/unmatched in current frame
        unmatch_ind_curr = np.where(kp_curr_land_id == -1)[0]
        
        # iterate through unmatched keypoints
        
        for ind in unmatch_ind_curr:
            [uL_curr, v_curr] = kp_curr[ind].pt
            Z = depth_imgs[i][int(v_curr), int(uL_curr)]
            if Z != 0:
                # add new landmark measurement to graph
                new_landmark_id = local_map.points.shape[0]
                kp_curr_land_id[ind] = new_landmark_id
                l = gtsam.symbol('l', new_landmark_id)
                kp_stereo = convert2stereoPt(uL_curr, v_curr, bf, Z)
                graph.add(gtsam.GenericStereoFactor3D(kp_stereo, stereo_model, x_curr,l, K_stereo))
                # add new landmarks to map
                init_curr_pose = gtsam.Pose3(gtsam.Rot3(R_init), t_init)
                Stereo_Camera = gtsam.StereoCamera(init_curr_pose, K_stereo)
                local_map.add_landmarks(np.expand_dims(Stereo_Camera.backproject(kp_stereo), axis=0))
                has_valid_measurement = True
            else:
                kp_curr_land_id[match.trainIdx] = -2

        if has_valid_measurement is True:   # check if current pose has any valid correspondences
            # add initial pose of current frame to map
            local_map.add_pose(R_init, np.expand_dims(t_init, axis=0), i)

    
        ## Create initial estimate for camera poses and landmarks ##
        initialEstimate = gtsam.Values()

        # initial estimates for all states
        for pose_ind in range(local_map.ts.shape[0]):
            initial_pose = gtsam.Pose3(gtsam.Rot3(local_map.Rs[pose_ind*3:pose_ind*3+3,:]), local_map.ts[pose_ind,:])
            x_n = gtsam.symbol('x',local_map.pose_ids[pose_ind]) 
            initialEstimate.insert(x_n, initial_pose)
        
        # initial estimates for all landmarks
        for land_ind in range(local_map.points.shape[0]):
            init_landmark_est = local_map.points[land_ind, :]
            l = gtsam.symbol('l', land_ind)
            initialEstimate.insert(l, init_landmark_est)

        # optimize
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        result = optimizer.optimize()

        # update map of stored poses and landmarks
        local_map.update_map(result)
        print("image: " + str(i) + " landmarks: " + str(local_map.points.shape[0]))

        # save current kp, descriptions and landmark ids for next frame
        kp_prev_land_id = kp_curr_land_id
        kp_prev = kp_curr
        des_prev = des_curr
    
    np.save('local_map_points',local_map.points)
    np.save('local_map_Rs',local_map.Rs)
    np.save('local_map_ts',local_map.ts)
    # np.save('pose_init_R', pose_init_R)
    # np.save('pose_init_t', pose_init_t)