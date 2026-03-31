# Human Pose Trajectory Prediction
## Demo

<img width="1920" height="941" alt="{60A32ED9-E8E2-4EC6-B99D-1536AE383597}" src="https://github.com/user-attachments/assets/6405521e-0670-4dd8-9d26-f4ca6bd0a226" />


## Overview
This project focuses on predicting human motion trajectories from a dance video using pose estimation and sequence modeling.

Human poses are first extracted frame-by-frame using MMPose, then transformed into temporal sequences to train predictive models.

## Pipeline
1. Pose extraction from video using MMPose (YOLOX + HRNet)
2. Preprocessing:
   - Savitzky-Golay smoothing
   - velocity computation
   - normalization (MinMax scaling)
3. Sequence construction:
   - input length: 30 frames
   - prediction horizon: 10 frames
4. Prediction models:
   - Linear Regression (baseline)
   - LSTM
   - GRU

## Results
Best model: **GRU**

- RMSE: **8.10 ± 1.74 px** (95% confidence interval, 10 runs)
- Stable performance across multiple runs
- Consistent improvement over linear baseline

## Files
- `Notebook_final_Lahbichi_Insafe.ipynb` : main notebook (full pipeline)
- `video_danse.mp4` : input video
- `results_video_danse.json` : pose estimation outputs (keypoints)
- `Rapport_MMPose.pdf` : detailed report
- `data_ready.npy` : processed dataset
- `data_coords_cleaned.npy` : cleaned coordinates

## Tech Stack
Python, PyTorch, MMPose, NumPy, scikit-learn

## Key Contributions
- End-to-end pipeline from raw video to trajectory prediction
- Temporal modeling of human motion using LSTM/GRU
- Quantitative evaluation with RMSE and confidence intervals
- Comparison with baseline model

## Report
A detailed report is available in:
- `Rapport_MMPose.pdf`
