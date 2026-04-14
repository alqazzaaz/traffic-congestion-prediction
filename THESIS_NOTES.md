# Thesis Notes & Justifications

## Chapter 4 – Data & Preprocessing

### Dataset
- Name: Urban Traffic Speed Dataset – Guangzhou, China
- Source: Zenodo DOI: 10.5281/zenodo.1205229
- Size: 1,855,572 observations after cleaning
- Roads: 214 urban segments
- Period: 2016-08-01 to 2016-09-30
- Interval: 10 minutes

### Outlier Removal
- Removed 17 rows with speed > 100 km/h
- Justification: sensor unreliability at extreme speeds
  in urban context, 0.0009% of data – negligible impact

### Congestion Threshold – 30 km/h
- International standards vary 20-30 km/h
  (Rao & Rao, 2012)
- China official standard: below 20 km/h
  (Su et al., 2019 – GJG[2008] 262)
- We chose 30 km/h because:
  1. Enables early warning before official
     congestion develops
  2. 20 km/h captures only 5.3% → extreme
     class imbalance, unsuitable for ML
  3. 30 km/h captures 18.6% → balanced
     and meaningful for binary classification
- Threshold analysis:
  Below 20 km/h: 5.3%
  Below 25 km/h: 10.2%
  Below 30 km/h: 18.6%
  Below 35 km/h: 33.4%
  Below 40 km/h: 52.5%

### EDA Key Findings
- Mean speed: 39.0 km/h
- Peak congestion: 18:00 weekday (59%)
- Weekend congestion lower and smoother
- Road 147 most congested (71.6%)
- Road 154 least congested (0.2%)
- Class distribution: 18.6% congestion,
  81.4% no congestion

### Preprocessing Pipeline

#### Sorting
- Data sorted chronologically by road_id, 
  date, time_id before all feature engineering

#### Feature Engineering
12 features used for modeling:

Time features:
- hour: extracted from start_time (0-23)
- weekday: day of week (0=Monday, 6=Sunday)
- is_weekend: binary flag (Saturday=5, Sunday=6)
- is_rush_hour: binary flag for hours 7-9 and 
  17-19 on weekdays
  → Validated: 36.6% congestion during rush hour
    vs 14.6% outside rush hour

Speed features:
- speed: current speed (km/h)
- speed_lag_1: speed 10 minutes ago
- speed_lag_2: speed 20 minutes ago
- speed_lag_3: speed 30 minutes ago
- speed_lag_6: speed 60 minutes ago
- speed_rolling_mean_3: average speed last 30 min
- speed_rolling_std_3: std speed last 30 min

Road feature:
- road_id: road segment identifier (1-214)

#### Target Variable
- congestion_t20: congestion 20 minutes ahead
- Binary: 0 = no congestion, 1 = congestion
- Created using shift(-2) per road segment
- Validated: row T congestion_t20 matches 
  row T+2 congestion

#### Feature Correlation with Target
Most predictive (negative – higher speed = less congestion):
- speed:               -0.642
- speed_rolling_mean:  -0.624
- speed_lag_1:         -0.612
- speed_lag_2:         -0.585
- speed_lag_3:         -0.559
- speed_lag_6:         -0.490

Positive correlation:
- is_rush_hour:        +0.224
- speed_rolling_std:   +0.219
- hour:                +0.206

Weak correlation:
- road_id:    -0.024
- weekday:    -0.030
- is_weekend: -0.053

#### Feature Multicollinearity
- Speed features highly correlated with each other
  (speed vs rolling_mean: 0.98, speed vs lag_1: 0.96)
- Acceptable for tree-based models and LSTM
- weekday vs is_weekend: 0.78 (expected)
- speed_rolling_std captures unique information
  (low correlation with other features)
- All 12 features retained

#### NaN Removal
- Removed 1,712 rows
- First 6 rows per road: no lag history available
- Last 2 rows per road: no future target available
- Remaining: 1,853,860 rows

#### Train / Validation / Test Split
- Method: chronological split (never random)
- Reason: time series data – future must not 
  leak into training
- Train:      70% = 1,297,702 rows
- Validation: 15% = 278,079 rows
- Test:       15% = 278,079 rows
- Congestion rates:
  Train: 20.0%, Validation: 14.9%, Test: 16.3%

#### Scaling
- Method: MinMaxScaler (range 0 to 1)
- Fitted on training data only
- Applied to validation and test
- Reason: required for LSTM, standardizes
  all features to equal scale

#### Class Imbalance Handling
- Method: class weights
- Training distribution: 20% congestion,
  80% no congestion
- Class weights mathematically derived 
  from training data distribution:
  No congestion (0): 0.6247
  Congestion (1):    2.5051
  Ratio: 4.0x
- Interpretation: missing a congestion case
  costs 4x more than missing a no-congestion case
- Applied to all three models:
  Random Forest: class_weight='balanced'
  XGBoost: scale_pos_weight=4.0
  LSTM: class_weight={0: 0.6247, 1: 2.5051}

### Saved Files
data/processed/X_train.csv  → 237.4 MB
data/processed/X_val.csv    →  51.1 MB
data/processed/X_test.csv   →  51.1 MB
data/processed/y_train.csv  →   6.2 MB
data/processed/y_val.csv    →   1.3 MB
data/processed/y_test.csv   →   1.3 MB
models/scaler.pkl            → MinMaxScaler
models/class_weights.pkl     → class weights

---

## Chapter 5 – Models

### Random Forest

#### Algorithm
- Ensemble of decision trees
- Each tree trained on random subset of data
  and features (bagging)
- Final prediction by majority vote

#### Hyperparameter Tuning
Three configurations tested on validation set:

| Config   | n_estimators | max_depth | F1   | Recall | Time  |
|----------|-------------|-----------|------|--------|-------|
| Baseline | 100         | None      | 0.69 | 0.60   | 61s   |
| Config 2 | 100         | 20        | 0.75 | 0.77   | 53s   |
| Config 3 | 200         | 20        | 0.75 | 0.77   | 114s  |

Winner: Config 2 (100 trees, max_depth=20)
Reason: Config 3 gives marginal improvement of 0.8%
        in recall at double the training cost.
        Config 2 is optimal efficiency/performance ratio.

Key finding: limiting max_depth improved recall from
60% to 77% – a 27.9% improvement – while reducing
training time from 61s to 53s.

#### Final Test Set Results (Config 2)
- Precision (congestion): 0.74
- Recall (congestion):    0.78
- F1 (congestion):        0.76
- Accuracy:               0.92
- AUC-ROC:                0.9585
- Training time:          53s
- Model size:             535.7 MB

#### Threshold Analysis (RF)
Default threshold 0.5 vs optimized:

| Threshold | Precision | Recall | F1    |
|-----------|-----------|--------|-------|
| 0.30      | 0.621     | 0.896  | 0.733 |
| 0.35      | 0.655     | 0.872  | 0.748 |
| 0.40      | 0.688     | 0.845  | 0.758 |
| 0.45      | 0.718     | 0.814  | 0.763 | ← best
| 0.50      | 0.744     | 0.779  | 0.761 |

Best threshold: 0.45
Best F1: 0.763
Recall improvement: 0.779 → 0.814 (+4.5%)
Justification: lower threshold catches more
congestion cases – preferred for early warning

#### Confusion Matrix (Test Set)
- True Negatives:  220,536
- False Positives:  12,155
- False Negatives:  10,028
- True Positives:   35,360

#### Feature Importance (Random Forest)
- speed:               0.282  ← most important
- speed_rolling_mean:  0.188
- speed_lag_1:         0.132
- speed_lag_2:         0.121
- speed_lag_3:         0.071
- speed_lag_6:         0.065
- speed_rolling_std:   0.043
- hour:                0.043
- road_id:             0.033
- weekday:             0.014
- is_rush_hour:        0.005
- is_weekend:          0.002

Key finding: speed features dominate. Time features
(is_rush_hour, is_weekend) have very low importance –
current speed implicitly encodes time-of-day patterns.

---

## Chapter 6 – Evaluation
(add results here)