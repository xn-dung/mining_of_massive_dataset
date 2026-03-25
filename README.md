# 🚕 Mining of Massive Datasets – Taxi Demand Prediction

This project is developed by **Group 9 – Class E22TTNT, PTIT University** as part of the *Mining of Massive Datasets* course.
## 📌 Overview

Urban transportation systems generate massive amounts of data every day. In this project, we aim to learn the **spatiotemporal distribution of taxi usage** from historical data in order to **predict future taxi demand**.

Our objective is to:
- 📍 Identify high-demand areas
- 🚗 Reduce idle vehicles in low-demand regions
- 🔄 Support intelligent taxi dispatching
- ⚡ Improve efficiency of urban mobility systems
## 🗂 Dataset

We use the **NYC Yellow Taxi dataset**, a large-scale real-world dataset:

- **Size**: ~28.5 GB  
- **Format**: Parquet  
- **Time Range**: 2011 → June 2024  

### 📊 Features include:
- Pickup & drop-off timestamps  
- Pickup & drop-off locations  
- Trip distance  
- Fare amount  
- Passenger count  
- Payment type  

### 🧠 Motivation
This dataset provides strong **temporal and spatial patterns**, making it ideal for demand prediction and large-scale data mining tasks.

## ⚙️ Pipeline

### 1. 📥 Data Collection
- Collect NYC Yellow Taxi data across multiple years
- Store data in Parquet format
### 2. 🧹 Data Preprocessing
- Handle missing values  
- Remove outliers:
  - Invalid distances or fares  
  - Unrealistic timestamps  
- Normalize features  
- Aggregate data into time intervals  
### 3. 🔄 Data Transformation
- Convert raw data into **time-series sequences**
- Generate:
  - Sequential features (for LSTM)
  - Spatial grid data (for CNN)
  - Static/topological features  

- Apply sliding window:
  - Input: past `T` timesteps  
  - Output: next timestep demand  
## 🧠 Models
To do this task, we experiment with **two separate models** ,compare their performance and improve them later

### 🔹 Model 1: CNN-based Model
- Learns **spatial patterns** from grid-based data  
- Captures local correlations between regions  
### 🔹 Model 2: LSTM-based Model
- Learns **temporal dependencies** in taxi demand  
- Models sequential patterns over time  
## 📊 Model Comparison

We compare both models based on:
- Prediction accuracy  
- Generalization ability  
- Stability during training  
## 📈 Evaluation Metrics

- SMAPE  
- RMSE  
- MAPE variants  
## 📚 Related Works

### 📄 DMVST-Net (2018)
🔗 https://arxiv.org/pdf/1802.08714

- Models spatial, temporal, and semantic dependencies  
- Captures interactions between regions and time evolution  

**Key idea:**  
Urban traffic demand is influenced by both spatial and temporal dynamics, requiring models that can capture both aspects effectively.

### 📄 Spatiotemporal Distribution Learning (2025)
🔗 https://arxiv.org/abs/2502.12213
- Focuses on learning **distribution patterns** instead of point predictions  
- Handles uncertainty in real-world data  

**Key idea:**  
Traffic demand follows a distribution rather than a fixed value, and modeling this distribution improves prediction robustness.
## 🧪 Implementation
- Sequence-based data sampling  
- Separate CNN and LSTM models  
- Custom loss function combining:
  - Relative error  
  - Mean Squared Error  

## 📅 Timeline

👉 Detailed timeline: https://docs.google.com/spreadsheets/d/1UW-vv7m2D7Uas-4YJzuEMECq9CMCWRZRFaTQ70r3FTw/edit?usp=sharing

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
