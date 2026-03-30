# Human Pose Trajectory Prediction

## Overview
This project studies human motion trajectory prediction from a dance video using pose estimation and recurrent neural networks.

Human poses are first extracted with MMPose, then processed to build temporal sequences for prediction.

## Pipeline
1. Pose extraction from video using MMPose (YOLOX + HRNet)
2. Preprocessing:
   - Savitzky-Golay smoothing
   - velocity computation
   - normalization
3. Prediction models:
   - Linear Regression
   - LSTM
   - GRU

## Results
Best model: **GRU**  
RMSE: **8.10 ± 1.74 px** (95% CI, 10 runs)

## Files
- `Notebook_final_Lahbichi_Insafe.ipynb` : main notebook
- `video_danse.mp4` : project demo video
- `Rapport_MMPose.pdf` : project report
- `data_ready.npy` : processed data
- `data_coords_cleaned.npy` : cleaned coordinates

## Tech Stack
Python, PyTorch, MMPose, NumPy, scikit-learn

## Report
A detailed report is available in:
- `Rapport_MMPose.pdf`