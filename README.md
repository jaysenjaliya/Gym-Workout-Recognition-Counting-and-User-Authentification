# Gym-Workout-Recognition-Counting-and-User-Authentification
<p align="center">
  <h1 align="center">🏋️ Gym Workout Recognition, Counting & User Authentication</h1>
  <p align="center">
    <em>Hybrid CNN-Dilated Self-Attention Model using Inertial & Body-Area Electrostatic Sensing</em>
  </p>
  <p align="center">
    <a href="#features"><img src="https://img.shields.io/badge/Activities-12_Classes-00A896?style=for-the-badge" alt="12 Classes"></a>
    <a href="#tech-stack"><img src="https://img.shields.io/badge/Framework-TensorFlow_2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
    <a href="#dataset"><img src="https://img.shields.io/badge/Dataset-RecGym-4285F4?style=for-the-badge" alt="RecGym"></a>
    <a href="#evaluation"><img src="https://img.shields.io/badge/Evaluation-Cross_User-8B5CF6?style=for-the-badge" alt="Cross-User"></a>
  </p>
</p>

---

## 📖 Project Overview

This project implements a **deep learning-based system** for automatic recognition and classification of **12 gym workout activities** using multi-modal wearable sensor data. The system leverages a **Hybrid CNN-Dilated Self-Attention** architecture that fuses **Inertial Measurement Unit (IMU)** signals with a novel **Human Body Capacitance (HBC)** sensing modality.

### 🔬 Key Innovation

Unlike traditional HAR systems that rely solely on accelerometer/gyroscope data, this project introduces **body-area electrostatic sensing (capacitance)** as a complementary modality. The dual-branch CNN architecture processes IMU and capacitance signals separately before fusing them, enabling the model to capture richer activity patterns.

### 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: Raw Sensor Data                       │
│              (Ax, Ay, Az, Gx, Gy, Gz, C₁)                     │
├──────────────────────┬──────────────────────────────────────────┤
│   IMU Branch (6ch)   │   Capacitance Branch (1ch)              │
│   ┌──────────────┐   │   ┌──────────────┐                     │
│   │ Conv2D + LN  │   │   │ Conv2D + LN  │                     │
│   │ DepthwiseConv│   │   │ DepthwiseConv│                     │
│   │ Conv2D + LN  │   │   │ Conv2D + LN  │                     │
│   └──────┬───────┘   │   └──────┬───────┘                     │
├──────────┴───────────┴──────────┴──────────────────────────────┤
│                    Concatenate (Post-Fusion)                    │
├────────────────────────────────────────────────────────────────┤
│              Sliding Window (n_windows=4)                       │
│   ┌──────────────────────────────────────────────┐             │
│   │  Multi-Head Self-Attention (4 heads, d=8)    │             │
│   │  ↓                                           │             │
│   │  Dilated TCN (dilation rates: 1, 2)          │             │
│   │  ↓                                           │             │
│   │  Dense → Activity Class                      │             │
│   └──────────────────────────────────────────────┘             │
│              Average Predictions Across Windows                 │
├────────────────────────────────────────────────────────────────┤
│                    Softmax → 12 Classes                         │
└────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **12-Class Activity Recognition** — Adductor, Arm Curl, Bench Press, Leg Curl, Leg Press, Null, Riding, Rope Skipping, Running, Squat, Stair Climber, Walking
- **Multi-Modal Sensor Fusion** — Combines 6-axis IMU (accelerometer + gyroscope) with body capacitance signals
- **Hybrid Deep Learning Architecture** — CNN for local feature extraction + Multi-Head Self-Attention for temporal dependencies + Dilated TCN for multi-scale patterns
- **Cross-User Evaluation** — Leave-One-Subject-Out (LOSO) cross-validation for robust generalization assessment
- **Class-Weighted Training** — Handles class imbalance through per-class sample weighting
- **Multiple Sensor Configurations** — Supports IMU-only, capacitance-only, and combined modes
- **Multiple Body Positions** — Data from wrist, leg, and pocket sensor placements

---

## 📦 Dataset

### RecGym Dataset

The **RecGym** dataset is a large-scale gym workout recognition benchmark:

| Property | Value |
|----------|-------|
| **Subjects** | 10 volunteers |
| **Sessions** | 5 per subject |
| **Activities** | 12 classes (11 exercises + Null) |
| **Sensor Positions** | Wrist, Leg, Pocket |
| **Sampling Rate** | 20 Hz |
| **Window Size** | 80 time-steps (4 seconds) |

### Sensor Channels

| Channel | Description |
|---------|-------------|
| `A_x`, `A_y`, `A_z` | Accelerometer (3-axis) |
| `G_x`, `G_y`, `G_z` | Gyroscope (3-axis) |
| `C_1` | Human Body Capacitance |


---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA support for accelerated training

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jaysenjaliya/Gym-Workout-Recognition-Counting-and-User-Authentification.git
   cd Gym-Workout-Recognition-Counting-and-User-Authentification
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux/macOS
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Option 1: Run the Full Training Pipeline

```bash
python main_TrainTest.py
```

This will:
- Load the RecGym dataset
- Train the model using LOSO cross-validation
- Save best model weights per subject
- Generate confusion matrices and performance logs

### Option 2: Use the Jupyter Notebook

```bash
jupyter notebook notebooks/Hybrid_CNN_Dilated_SelfAttention.ipynb
```

The notebook provides a self-contained, step-by-step walkthrough:
1. **Imports & Setup** — Load all dependencies
2. **Data Preprocessing** — Load, filter, window, and normalize data
3. **Model Definition** — Build the Hybrid CNN-Dilated Self-Attention model
4. **Cross-User Training** — Train on User 1, test on User 2 (and vice versa)
5. **Evaluation** — Accuracy, Macro F1, Cohen's Kappa, confusion matrices

### Configuration

Key hyperparameters in `main_TrainTest.py`:

---

## 📈 Evaluation

### Cross-User Protocol

The model is evaluated using **Leave-One-Subject-Out (LOSO)** cross-validation:
- For each fold, one subject is held out as the test set
- The remaining subjects are used for training
- This tests **generalization to unseen users** — the most rigorous evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **Macro F1-Score** | Unweighted mean of per-class F1 scores |
| **Cohen's Kappa** | Agreement metric accounting for chance |
| **Confusion Matrix** | Per-class prediction analysis |



<p align="center">
  <em>Built with ❤️ for fitness and deep learning</em>
</p>
