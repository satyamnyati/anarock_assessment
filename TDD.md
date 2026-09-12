# Technical Design Document (TDD)
## Resident Facility Booking & Nudge Prediction Engine

---

## 1. System Context & Business Goals

Residential management platforms (such as ANACITY / ANAROCK) oversee community shared amenities including Gyms, Swimming Pools, Tennis Courts, Badminton Courts, and Clubhouses. Residents exhibit distinct periodic usage routines and varying advance reservation habits ("lead times").

### Primary Objective
Build an automated predictive engine that analyzes historical resident booking logs to forecast four target outputs for future booking events:
1. **Facility (`next_facility`)**: Target amenity choice.
2. **Usage Day (`next_usage_day`)**: Day of week (Mon-Sun).
3. **Usage Hour (`next_usage_hour`)**: Time of day (0-23 hours).
4. **Nudge Timestamp ($\hat{T}_{\text{nudge}}$)**: Recommended notification delivery time derived from predicted lead time $\hat{\Delta t}_{\text{lead}}$.

$$\hat{T}_{\text{nudge}} = \hat{T}_{\text{usage}} - \hat{\Delta t}_{\text{lead}}$$

---

## 2. Synthetic Data Generator Design

To simulate realistic resident activity, the data generator models heterogeneous behavioral personas over a multi-month period (120 residents across 120 simulated days).

### Persona Specifications & Behavioral Profiles

| Persona Type | Facility Preference Distribution | Day-of-Week Schedule | Hour Preference | Lead Time Distribution ($\Delta t$) |
| :--- | :--- | :--- | :--- | :--- |
| **Gym Regular** | Gym (90%), Pool (10%) | Mon, Wed, Fri | 07:00 (85%) / 18:00 (15%) | Normal($\mu = 3.0\text{h}$, $\sigma = 0.5\text{h}$) |
| **Racket Enthusiast** | Badminton / Tennis (Uniform) | Tue, Thu, Sat | 19:00 (90%) / 20:00 (10%) | Normal($\mu = 24.0\text{h}$, $\sigma = 2.0\text{h}$) |
| **Weekend Pool Goer** | Pool (85%), Clubhouse (15%) | Sat, Sun | 15:00 (80%) / 16:00 (20%) | Normal($\mu = 6.0\text{h}$, $\sigma = 1.0\text{h}$) |
| **Casual User** | All Facilities (Uniform) | Stochastic (~15% daily prob) | Uniform(08:00 to 20:00) | Normal($\mu = 12.0\text{h}$, $\sigma = 3.0\text{h}$) |

### Timestamps & Sequence Generation Logic
For each generated booking record $i$:
* $\text{usage\_timestamp}_i = \text{date} + \text{hour}_i$
* $\text{lead\_time\_hours}_i = \max(0.5, \text{sample\_lead\_time})$
* $\text{booking\_timestamp}_i = \text{usage\_timestamp}_i - \text{lead\_time\_hours}_i$

---

## 3. Feature Pipeline & Data Leakage Prevention Controls

### Leakage Prevention Strategy
To guarantee zero future-data leakage during time-series inference:
1. Events are ordered strictly by resident and chronological usage timestamp $t_1 < t_2 < \dots < t_N$.
2. Features for event $k$ at time $t_k$ are derived strictly using records $\{1, 2, \dots, k-1\}$.
3. The shift operator ($\text{shift}(1)$) is applied across all grouped aggregations.

### Engineered Feature Set

1. **Lagged Categorical States**:
   * `prev_facility_1`: Facility booked in event $k-1$.
   * `prev_facility_2`: Facility booked in event $k-2$.
   * `prev_usage_day`: Day of week of event $k-1$.
   * `prev_usage_hour`: Hour of day of event $k-1$.

2. **Expanding Historical Ratio Features**:
   Calculated per resident $r$ over all prior events up to $k-1$:
   $$\text{hist\_ratio}_{\text{fac}} = \frac{\sum_{j=1}^{k-1} \mathbf{1}(\text{facility}_j = \text{fac})}{k - 1}$$
   $$\text{hist\_day\_ratio}_{\text{day}} = \frac{\sum_{j=1}^{k-1} \mathbf{1}(\text{day}_j = \text{day})}{k - 1}$$

3. **Cumulative Lead Time Mean**:
   $$\text{cum\_lead\_time\_mean} = \frac{1}{k - 1} \sum_{j=1}^{k-1} \text{lead\_time}_j$$

4. **Categorical Feature Encoding**:
   * One-hot dummy columns are generated for categorical lag variables.
   * Original string columns (`prev_facility_1`, `prev_facility_2`, `prev_usage_day`) are retained in the base dataframe to enable detailed prediction review output formatting while being excluded from the model training input matrix $X$.

---

## 4. Modeling Architecture & Nudge Logic

```
               ┌──────────────────────────────┐
               │   Engineered Feature Set X   │
               └──────────────┬───────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Random Forest   │ │  Random Forest   │ │  Random Forest   │
│ (Facility Model) │ │   (Day Model)    │ │   (Hour Model)   │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   Predicted Fac       Predicted Day       Predicted Hour
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │ Predicted Usage Timestamp T  │
               └──────────────┬───────────────┘
                              │
                              │ ┌──────────────────────────────┐
                              │ │  Gradient Boosting Regressor │
                              │ │      (Lead Time Model)       │
                              │ └──────────────┬───────────────┘
                              │                │
                              ▼                ▼
                       Predicted Lead Time (\Delta t)
                              │
                              ▼
               ┌──────────────────────────────┐
               │   Predicted Nudge Timestamp  │
               │    (T_nudge = T_use - \Delta t)│
               └──────────────────────────────┘
```

### Models & Target Mapping
* **Facility Predictor**: Multi-class Random Forest Classifier ($K=5$ facilities).
* **Usage Day Predictor**: Multi-class Random Forest Classifier ($K=7$ days).
* **Usage Hour Predictor**: Multi-class Random Forest Classifier ($K=24$ hours).
* **Lead Time Predictor**: Gradient Boosting Regressor (Continuous $\Delta t$ in hours).

### Chronological Holdout Split
* **Train Set**: First 60% of temporal timeline.
* **Validation Set**: Next 20% of temporal timeline.
* **Test Holdout Set**: Final 20% of temporal timeline.
* Records with zero historical background (`hist_count == 0`) are filtered out during holdout evaluation.

---

## 5. Output Artifacts & Verification Specifications

### 1. Model Artifact (`artifacts/models.pkl`)
Serialized dictionary containing:
* `clf_facility`: Trained Random Forest model for facility prediction.
* `clf_day`: Trained Random Forest model for day prediction.
* `clf_hour`: Trained Random Forest model for hour prediction.
* `reg_lead`: Trained Gradient Boosting Regressor for lead time prediction.
* `feature_cols`: Ordered list of input feature column names.

### 2. Evaluation Metrics Schema (`artifacts/metrics.json`)
```json
{
    "Facility Accuracy": 0.6322,
    "Usage Day Accuracy": 0.7531,
    "Usage Hour Accuracy": 0.7028,
    "Lead Time MAE (Hours)": 1.2242
}
```

### 3. Review Table Format (`output/prediction_review_output.csv`)
Columns:
* `Resident Reference`: Unique resident string (e.g., `R-004`).
* `PAST BOOKING CONTEXT`: Historical lag-1 state (`Facility/Day/Hour`).
* `PREDICTION`: Combined prediction string formatted as `Facility/Day/Hour | Nudge Day/Time`.
* `ACTUAL`: Ground truth string formatted as `Facility/Day/Hour | Booked Day/Time`.
* `MATCH`: Score indicator string, e.g., `YES (4 of 4)` or `NO (3 of 4)`.

---

## 6. Limitations & Future Enhancements

1. **Cold-Start Latency**: First-time residents lacking historical bookings resort to default global population averages until sufficient events ($k \ge 3$) accumulate.
2. **Abrupt Persona Drift**: Sudden life routine changes (e.g., shifting shift times from morning to evening) require $2-3$ events for expanding historical ratios to re-adapt.
3. **Multi-Facility Nudge Constraints**: Nudge delivery timing currently assumes unconstrained facility capacity. Incorporating real-time slot availability constraints as external model inputs is a recommended future optimization.