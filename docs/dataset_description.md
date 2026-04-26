# RecGym Dataset Documentation

## Overview

The **RecGym** dataset is a comprehensive benchmark for gym workout recognition, collected using multi-modal wearable sensors. It captures 12 gym exercise types performed by 10 volunteers across 5 sessions each.

📥 **Download:** [Kaggle — RecGym Dataset](https://www.kaggle.com/datasets/zhaxidelebsz/10-gym-exercises-with-615-abstracted-features)

📄 **Homepage:** [https://zhaxidele.github.io/RecGym/](https://zhaxidele.github.io/RecGym/)

---

## Dataset Statistics

| Property | Value |
|----------|-------|
| **Subjects** | 10 volunteers |
| **Sessions per subject** | 5 |
| **Total activity classes** | 12 (11 exercises + 1 Null) |
| **Sensor positions** | Wrist, Leg, Pocket |
| **Sampling rate** | 20 Hz |
| **Sensor channels** | 7 (6 IMU + 1 HBC) |
| **File format** | CSV (`RecGym.csv`, ~475 MB) |

---

## Activity Classes

| ID | Activity | Description |
|----|----------|-------------|
| 1 | Adductor | Inner thigh adductor machine exercise |
| 2 | Arm Curl | Bicep curl with dumbbells or machine |
| 3 | Bench Press | Chest press on a flat bench |
| 4 | Leg Curl | Hamstring curl on machine |
| 5 | Leg Press | Quadriceps press on machine |
| 6 | Null | No activity / rest / transition |
| 7 | Riding | Stationary bike cycling |
| 8 | Rope Skipping | Jump rope exercise |
| 9 | Running | Treadmill running |
| 10 | Squat | Bodyweight or weighted squats |
| 11 | Stair Climber | Stair climbing machine |
| 12 | Walking | Treadmill walking |

---

## Sensor Channels

### IMU (Inertial Measurement Unit) — 6 channels

| Channel | Description | Unit |
|---------|-------------|------|
| `A_x` | Accelerometer X-axis | m/s² |
| `A_y` | Accelerometer Y-axis | m/s² |
| `A_z` | Accelerometer Z-axis | m/s² |
| `G_x` | Gyroscope X-axis | rad/s |
| `G_y` | Gyroscope Y-axis | rad/s |
| `G_z` | Gyroscope Z-axis | rad/s |

### HBC (Human Body Capacitance) — 1 channel

| Channel | Description | Unit |
|---------|-------------|------|
| `C_1` | Body capacitance signal | pF (picofarads) |

> **Why HBC?** Body capacitance changes when different body postures interact with the surrounding electric field. This provides complementary information to IMU — especially useful for distinguishing exercises with similar motion patterns but different body configurations (e.g., Leg Curl vs. Leg Press).

---

## CSV Column Structure

The `RecGym.csv` file contains the following columns:

| Column | Description |
|--------|-------------|
| `Object` / `Subject` | Subject ID (1–10) |
| `Day` | Session/day number |
| `LocalSerie` | Local series identifier |
| `Win_Num` | Window number |
| `Win_In_Num` | Window internal number |
| `Workout` | Activity label (string) |
| `Workout_time` | Timestamp within workout |
| `Position` | Sensor position (wrist / leg / pocket) |
| `A_x` | Accelerometer X |
| `A_y` | Accelerometer Y |
| `A_z` | Accelerometer Z |
| `G_x` | Gyroscope X |
| `G_y` | Gyroscope Y |
| `G_z` | Gyroscope Z |
| `C_1` | Body capacitance |

---

## Data Preprocessing Pipeline

### 1. Filtering
- Select a specific sensor position (e.g., **wrist**)
- Select subjects for train/test split

### 2. Feature Selection
Three sensor modes are supported:

| Mode | Channels | Count |
|------|----------|-------|
| `combine` | A_x, A_y, A_z, G_x, G_y, G_z, C_1 | 7 |
| `imu` | A_x, A_y, A_z, G_x, G_y, G_z | 6 |
| `cap` | C_1 | 1 |

### 3. Windowing
- **Window size:** 80 time-steps (4 seconds at 20 Hz)
- **Stride:** 40 time-steps (50% overlap) for cross-user mode
- **Label assignment:** Majority vote within each window

### 4. Normalization
- `StandardScaler` fitted on training data, applied to both train and test

### 5. Reshaping
- Final shape: `(N, 1, 80, channels)` for CNN input
- Labels: One-hot encoded with `num_classes=12`

---

## Evaluation Protocol

### Leave-One-Subject-Out (LOSO)
- **10 folds:** Each fold uses 1 subject as test set, remaining 9 for training
- Tests **cross-user generalization** — the model has never seen the test user

### Cross-User (2-User)
- **Simplified protocol:** Train on User 1, test on User 2 (and vice versa)
- Used in the notebook for rapid experimentation

---

## Usage in Code

```python
from data_loader_RecGym import RecGym_DATA, load_data

# Load dataset with windowing
root_path = "."  # directory containing RecGym.csv
dataloader = load_data(
    root_path=root_path,
    batch_size=64,
    window_size=4,      # 4 seconds
    overlap_size=2       # 2 seconds overlap
)

# Iterate over batches
for samples, labels in dataloader:
    print(f"Batch shape: {samples.shape}")  # (64, 80, 7)
    print(f"Labels shape: {labels.shape}")  # (64,)
    break
```

---

## Citation

If you use the RecGym dataset, please cite the original paper:

```bibtex
@article{recgym,
  title={Hybrid CNN-Dilated Self-Attention Model Using Inertial and Body-Area 
         Electrostatic Sensing for Gym Workout Recognition, Counting and 
         User Authentication},
  author={Zhaxidele et al.},
  year={2022}
}
```

---

## Notes

- The `RecGym.csv` file (~475 MB) is **not included** in this repository due to size constraints
- Download it from the Kaggle link above and place it in the project root directory
- The dataset is pre-segmented into abstracted features — raw sensor data at higher sampling rates may exist separately
