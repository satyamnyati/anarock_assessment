import json
import os
import pickle

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
NUM_RESIDENTS = 120
DAYS_SIMULATED = 120
START_DATE = datetime(2026, 1, 1, 0, 0, 0)
FACILITIES = ["Gym", "Pool", "Badminton", "Tennis", "Clubhouse"]
DAYS_MAP = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def generate_synthetic_data():
    records = []
    personas = {
        f"R-{r:03d}": np.random.choice(["gym_regular", "racket_enthusiast", "weekend_pool", "casual"])
        for r in range(1, NUM_RESIDENTS + 1)
    }

    for r_id, p_type in personas.items():
        for day_i in range(DAYS_SIMULATED):
            current_day_dt = START_DATE + timedelta(days=day_i)
            dow = current_day_dt.weekday()  # 0 = Mon, 6 = Sun
            
            will_book = False
            if p_type == "gym_regular" and dow in [0, 2, 4]:  # Mon, Wed, Fri
                will_book = np.random.rand() > 0.10
                facility = "Gym" if np.random.rand() > 0.10 else "Pool"
                hour = 7 if np.random.rand() > 0.15 else 18
                lead_hours = float(np.random.normal(3.0, 0.5))
            elif p_type == "racket_enthusiast" and dow in [1, 3, 5]:  # Tue, Thu, Sat
                will_book = np.random.rand() > 0.15
                facility = np.random.choice(["Tennis", "Badminton"])
                hour = 19 if np.random.rand() > 0.10 else 20
                lead_hours = float(np.random.normal(24.0, 2.0))
            elif p_type == "weekend_pool" and dow in [5, 6]:  # Sat, Sun
                will_book = np.random.rand() > 0.10
                facility = "Pool" if np.random.rand() > 0.15 else "Clubhouse"
                hour = 15 if np.random.rand() > 0.20 else 16
                lead_hours = float(np.random.normal(6.0, 1.0))
            elif p_type == "casual" and np.random.rand() < 0.15:  # Random periodic usage
                will_book = True
                facility = np.random.choice(FACILITIES)
                hour = int(np.random.choice(range(8, 20)))
                lead_hours = float(np.random.normal(12.0, 3.0))

            if will_book:
                usage_dt = current_day_dt.replace(hour=hour, minute=0, second=0)
                lead_hours = max(0.5, lead_hours)
                booking_dt = usage_dt - timedelta(hours=lead_hours)

                records.append({
                    "resident_id": r_id,
                    "facility_id": facility,
                    "booking_timestamp": booking_dt,
                    "usage_timestamp": usage_dt,
                    "usage_day": DAYS_MAP[usage_dt.weekday()],
                    "usage_hour": usage_dt.hour,
                    "lead_time_hours": lead_hours
                })

    return pd.DataFrame(records).sort_values(by="usage_timestamp").reset_index(drop=True)


def build_features(df):
    df = df.sort_values(by=["resident_id", "usage_timestamp"]).reset_index(drop=True)
    
    # Lagged features
    df["prev_facility_1"] = df.groupby("resident_id")["facility_id"].shift(1).fillna("None")
    df["prev_facility_2"] = df.groupby("resident_id")["facility_id"].shift(2).fillna("None")
    df["prev_usage_day"] = df.groupby("resident_id")["usage_day"].shift(1).fillna("None")
    df["prev_usage_hour"] = df.groupby("resident_id")["usage_hour"].shift(1).fillna(-1)
    
    df["hist_count"] = df.groupby("resident_id").cumcount()
    
    # Historical facility preference ratios per resident
    for fac in FACILITIES:
        df[f"hist_ratio_{fac}"] = (
            df.groupby("resident_id")["facility_id"]
            .transform(lambda s: (s.shift(1) == fac).cumsum() / np.maximum(1, df.groupby("resident_id").cumcount()))
        )

    # Historical day-of-week preference ratios per resident
    for d in DAYS_MAP:
        df[f"hist_day_ratio_{d}"] = (
            df.groupby("resident_id")["usage_day"]
            .transform(lambda s: (s.shift(1) == d).cumsum() / np.maximum(1, df.groupby("resident_id").cumcount()))
        )

    # Expanding historical mean lead time
    df["cum_lead_time_mean"] = (
        df.groupby("resident_id")["lead_time_hours"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(12.0)
    )

    # Categorical one-hot encoding
    cat_cols = ["prev_facility_1", "prev_facility_2", "prev_usage_day"]
    dummies = pd.get_dummies(df[cat_cols], drop_first=False)
    df_encoded = pd.concat([df, dummies], axis=1)
    
    return df_encoded


def main():
  os.makedirs("artifacts", exist_ok=True)
  os.makedirs("output", exist_ok=True)

  print("Generating Synthetic Dataset...")
  raw_df = generate_synthetic_data()
  raw_df.to_csv("output/synthetic_facility_bookings.csv", index=False)
  print(f"Total synthetic records generated: {len(raw_df)}")

  df_features = build_features(raw_df)

  n = len(df_features)
  train_end = int(n * 0.6)
  val_end = int(n * 0.8)

  train_df = df_features.iloc[:train_end]
  test_df = df_features.iloc[val_end:].copy()

  test_df = test_df[test_df["hist_count"] > 0].reset_index(drop=True)

  # Exclude non-feature metadata and raw categorical strings from model training features
  non_feature_cols = [
      "resident_id",
      "facility_id",
      "booking_timestamp",
      "usage_timestamp",
      "usage_day",
      "usage_hour",
      "lead_time_hours",
      "prev_facility_1",
      "prev_facility_2",
      "prev_usage_day",
  ]
  feature_cols = [c for c in df_features.columns if c not in non_feature_cols]

  X_train = train_df[feature_cols]
  y_train_facility = train_df["facility_id"]
  y_train_day = train_df["usage_day"]
  y_train_hour = train_df["usage_hour"]
  y_train_lead = train_df["lead_time_hours"]

  X_test = test_df[feature_cols]
  y_test_facility = test_df["facility_id"]
  y_test_day = test_df["usage_day"]
  y_test_hour = test_df["usage_hour"]
  y_test_lead = test_df["lead_time_hours"]

  print("Training Multi-Output Models...")
  clf_facility = RandomForestClassifier(
      n_estimators=100, random_state=SEED
  ).fit(X_train, y_train_facility)
  clf_day = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(
      X_train, y_train_day
  )
  clf_hour = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(
      X_train, y_train_hour
  )
  reg_lead = GradientBoostingRegressor(
      n_estimators=100, random_state=SEED
  ).fit(X_train, y_train_lead)

  pred_facility = clf_facility.predict(X_test)
  pred_day = clf_day.predict(X_test)
  pred_hour = clf_hour.predict(X_test)
  pred_lead = reg_lead.predict(X_test)

  acc_fac = accuracy_score(y_test_facility, pred_facility)
  acc_day = accuracy_score(y_test_day, pred_day)
  acc_hour = accuracy_score(y_test_hour, pred_hour)
  mae_lead = mean_absolute_error(y_test_lead, pred_lead)

  metrics = {
      "Facility Accuracy": round(float(acc_fac), 4),
      "Usage Day Accuracy": round(float(acc_day), 4),
      "Usage Hour Accuracy": round(float(acc_hour), 4),
      "Lead Time MAE (Hours)": round(float(mae_lead), 4),
  }

  print("\n--- Evaluation Metrics on Chronological Holdout ---")
  for k, v in metrics.items():
    print(f"{k}: {v}")

  with open("artifacts/models.pkl", "wb") as f:
    pickle.dump({
        "clf_facility": clf_facility,
        "clf_day": clf_day,
        "clf_hour": clf_hour,
        "reg_lead": reg_lead,
        "feature_cols": feature_cols,
    }, f)

  with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

  review_rows = []
  for idx, row in test_df.iterrows():
    r_id = row["resident_id"]

    past_fac = row["prev_facility_1"]
    past_day = row["prev_usage_day"]
    past_hr = (
        int(row["prev_usage_hour"]) if row["prev_usage_hour"] != -1 else 0
    )
    past_str = f"{past_fac}/{past_day}/{past_hr:02d}:00"

    act_fac = row["facility_id"]
    act_day = row["usage_day"]
    act_hr = row["usage_hour"]
    act_lead = row["lead_time_hours"]
    act_booking_dt = row["booking_timestamp"]
    act_nudge_str = f"Booked {DAYS_MAP[act_booking_dt.weekday()]}/{act_booking_dt.strftime('%H:%M')}"
    actual_str = f"{act_fac}/{act_day}/{act_hr:02d}:00 | {act_nudge_str}"

    p_fac = pred_facility[idx]
    p_day = pred_day[idx]
    p_hr = pred_hour[idx]
    p_lead = pred_lead[idx]

    pred_usage_dt = row["usage_timestamp"].replace(hour=p_hr, minute=0)
    pred_nudge_dt = pred_usage_dt - timedelta(hours=p_lead)
    pred_nudge_str = f"Nudge {DAYS_MAP[pred_nudge_dt.weekday()]}/{pred_nudge_dt.strftime('%H:%M')}"
    prediction_str = f"{p_fac}/{p_day}/{p_hr:02d}:00 | {pred_nudge_str}"

    m1 = p_fac == act_fac
    m2 = p_day == act_day
    m3 = p_hr == act_hr
    m4 = abs(p_lead - act_lead) <= 3.0

    matches_count = sum([m1, m2, m3, m4])
    match_status = "YES" if matches_count == 4 else "NO"
    score_str = f"{match_status} ({matches_count} of 4)"

    review_rows.append({
        "Resident Reference": r_id,
        "PAST BOOKING CONTEXT": past_str,
        "PREDICTION (facility/day/use time / nudge time)": prediction_str,
        "ACTUAL (facility/day/use time / booked at)": actual_str,
        "MATCH": score_str,
    })

  review_df = pd.DataFrame(review_rows)
  review_df.to_csv("output/prediction_review_output.csv", index=False)
  print("\nPrediction review output saved to 'output/prediction_review_output.csv'.")


if __name__ == "__main__":
  main()