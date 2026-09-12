# Resident Facility Booking & Nudge Prediction Engine

An end-to-end Machine Learning pipeline developed for predicting resident amenity usage patterns and optimal notification ("nudge") delivery times. The system generates realistic synthetic booking data, constructs leakage-safe temporal and behavioral profile features, trains multi-output predictive models (predicting facility, day of week, hour of day, and advance booking lead time), and evaluates performance on a chronologically separated test set.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Environment Setup & Installation](#environment-setup--installation)
- [Dependencies](#dependencies)
- [Running the Pipeline](#running-the-pipeline)
- [Output Validation Artifacts](#output-validation-artifacts)

---

## Project Overview

In residential communities managed by platforms like ANACITY / ANAROCK, residents share facilities such as Gyms, Swimming Pools, Tennis Courts, Badminton Courts, and Clubhouses. Anticipating when a resident is likely to use a facility allows the community management platform to send timely nudges or notifications, improving facility utilization and resident engagement.

This project implements:
1. **Synthetic Data Generator**: Simulates realistic historical facility booking events with heterogenous resident personas (Gym Regulars, Racket Enthusiasts, Weekend Pool Goers, Casual Users) with structured weekly routines and lead-time distributions.
2. **Leakage-Safe Feature Pipeline**: Constructs lagged prior event features, expanding historical preference ratios, and cumulative lead time statistics calculated strictly prior to each event.
3. **Multi-Output Model Suite**: Trains Random Forest classifiers for target categorical outputs (`facility`, `day`, `hour`) and a Gradient Boosting Regressor for continuous lead time ($\Delta t$).
4. **Nudge Delivery Scheduler**: Computes predicted nudge timestamps ($\hat{T}_{\text{nudge}} = \hat{T}_{\text{usage}} - \hat{\Delta t}_{\text{lead}}$).
5. **Prediction Review Generator**: Exports detailed record-by-record comparison outputs matching actual usage against predicted values.

---

## Directory Structure

```text
anarock_assessment/
├── README.md                         # Project documentation and setup guide
├── TDD.md                            # Technical Design Document (Architecture & Specs)
├── pipeline.py                       # Self-contained runnable Python workflow
├── requirements.txt                  # Python dependencies
├── artifacts/
│   ├── metrics.json                  # Output evaluation metrics (Accuracy, MAE)
│   └── models.pkl                    # Serialized model estimators and metadata
└── output/
    ├── prediction_review_output.csv  # Record-by-record prediction comparison table
    └── synthetic_facility_bookings.csv # Raw synthetic dataset
```

---

## Environment Setup & Installation

### Prerequisites
* **Python**: Version 3.9 or higher (Python 3.11 recommended)
* **Virtual Environment Tool**: `venv` (standard library) or `conda`

### Step-by-Step Setup Instructions

1. **Clone or Download the Project Repository**:
   Navigate to your local working directory:
   ```bash
   cd anarock_assessment
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   * **Linux / macOS**:
     ```bash
     source .venv/bin/activate
     ```
   * **Windows (Command Prompt / PowerShell)**:
     ```cmd
     .venv\Scripts\activate
     ```

4. **Upgrade Package Installer**:
   ```bash
   pip install --upgrade pip
   ```

5. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dependencies

The project relies on standard Python data science libraries specified in `requirements.txt`:

| Package | Minimum Version | Purpose |
| :--- | :--- | :--- |
| `numpy` | `1.24.0` | Numerical calculations & random distribution sampling |
| `pandas` | `2.0.0` | Data manipulation, feature engineering, and CSV output |
| `scikit-learn` | `1.2.0` | Model training (RandomForest, GradientBoosting) and evaluation |

---

## Running the Pipeline

To execute the end-to-end workflow (data generation, feature engineering, chronological train/test split, model training, evaluation, and report generation), run:

```bash
python pipeline.py
```

### Expected Execution Output

When executed successfully, the script prints summary information to stdout:

```text
Generating Synthetic Dataset...
Total synthetic records generated: 5183
Training Multi-Output Models...

--- Evaluation Metrics on Chronological Holdout ---
Facility Accuracy: 0.7952
Usage Day Accuracy: 0.8618
Usage Hour Accuracy: 0.7903
Lead Time MAE (Hours): 0.7837

Prediction review output saved to 'output/prediction_review_output.csv'.
```

---

## Output Validation Artifacts

Upon running `pipeline.py`, the following output files are updated and available for review:

1. **`output/synthetic_facility_bookings.csv`**
   * Raw historical booking logs generated by the synthetic engine.
   * Fields: `resident_id`, `facility_id`, `booking_timestamp`, `usage_timestamp`, `usage_day`, `usage_hour`, `lead_time_hours`.

2. **`output/prediction_review_output.csv`**
   * Review table comparing actual test data against predicted target variables for each test record.
   * Format includes:
     * `Resident Reference`: Identifier (e.g., `R-004`).
     * `PAST BOOKING CONTEXT`: Previous facility, day, and usage hour.
     * `PREDICTION`: Predicted facility/day/hour and derived nudge time.
     * `ACTUAL`: Actual ground truth facility/day/hour and actual booking timestamp.
     * `MATCH`: Score indicating overall match (e.g., `YES (4 of 4)` or `NO (3 of 4)`).

3. **`artifacts/metrics.json`**
   * Structured JSON containing model performance figures across test data.

4. **`artifacts/models.pkl`**
   * Pickled Python dictionary containing trained classifier objects (`clf_facility`, `clf_day`, `clf_hour`, `reg_lead`) and feature column ordering for reproducible inference.
```
