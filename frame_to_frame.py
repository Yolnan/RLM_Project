import numpy as np
import cv2
from matplotlib import pyplot as plt
import open3d as o3d


def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def essentialMatrix(F):
    # Camera Intrinsics matrix
    K1 = np.array([[5.533317252305629381e+02, 0.0, 3.286542425444330320e+02],[0.0, 5.507826069073726103e+02, 2.520388973352311268e+02],[0.0, 0.0, 1.000000000000000000e+00]])
    K2 = np.array([[5.533317252305629381e+02, 0.0, 3.286542425444330320e+02],[0.0, 5.507826069073726103e+02, 2.520388973352311268e+02],[0.0, 0.0, 1.000000000000000000e+00]])
    E = K2.T @ F @ K1
    U,S,Vh = np.linalg.svd(E)
    E = E/S[0]
    # E = E/E[2,2]  # normalizing, had divide by zero issues
    return E


def compute_F(img1,img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    return F

# def computeRt(E):
#     U,S,Vt = np.linalg.svd(E)
#     W = np.array([[0,-1,0],[1, 0, 0],[0,0,1]])
#     Winv = W.T
#     Rot = U @ Winv @ Vt
#     translation = U @ W @ S @ (U.T)
#     return Rot, translation

def computeRt(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    K = np.array([[5.533317252305629381e+02, 0.0, 3.286542425444330320e+02],[0.0, 5.507826069073726103e+02, 2.520388973352311268e+02],[0.0, 0.0, 1.000000000000000000e+00]])
    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K)
    pts, R, t, mask = cv2.recoverPose(E, pts1, pts2)
    return R, np.squeeze(t,axis=1)

def computeRt(source_color, target_color, source_depth, target_depth, K):
    # create pinhole camera intrinsic object w/ width, height, fx, fy, cx, cy
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(source_color.shape[1], source_color.shape[0],
        K[0,0], K[1,1], K[0,2], K[1,2])

    source_color = o3d.geometry.Image(source_color)
    target_color = o3d.geometry.Image(target_color)
    source_depth = o3d.geometry.Image(source_depth)
    target_depth = o3d.geometry.Image(target_depth)
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(target_color, target_depth)
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)

    [success_hybrid_term, trans_hybrid_term,info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    return trans_hybrid_term[0:3, 0:3], trans_hybrid_term[0:3,3]