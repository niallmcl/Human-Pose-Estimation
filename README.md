# Human-Pose-Estimation
This project implements real-time human pose estimation on the CPU or GPU.

Each person's bounding box is first estimated using open-CV. The bounding box is then passed to a CNN which performs pose estimation in 2D and 3D. The CNN was trained using multi-task training, on the H36M and MPII human pose datasets to perform both 3D and 2D human pose estimation.

This code will perform pose estimation in 'real-time'  ~15fps on an Intel i7 laptop CPU. The bottleneck is the bounding box estimation, rather than the CNN.

# Requirements
Pytorch
OpenCV

Note - this code was written for Pytorch 0.4
