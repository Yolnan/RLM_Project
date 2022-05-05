CMU RLM 16833 Spring 2022
Team Members: Yolnan Chen, Sagar Sachdev, Troy Vicsik

Software Used
OS: Ubuntu 20.04 LTS

Python: Conda virtual environment w/ Python 3.9.7

Python Libraries:
-opencv 4.5.3
-numpy 1.21.4
-matplotlib 3.4.3
-glob
-open3d 0.15.2
-gtsam 4.1.1

Python Scripts:
-main.py; used to generate estimate trajectories
-frame_to_frame.py; functions used to calculate frame to frame transformation
-map.py; class definition of map object used to store poses and landmarks
-map_viewer.py; generates plots for results

Other files:
-absolute pose/*.txt; absolute poses stored as txt files
-*.npy; saved trajectory estimates
-KeyFrameTrajectory_corner0.txt; ORB-SLAM3 Stereo D435i camera trajectory using corner0 VOID dataset

Dataset repo link: https://github.com/alexklwong/void-dataset

ORB-SLAM3 repo link: https://github.com/UZ-SLAMLab/ORB_SLAM3 