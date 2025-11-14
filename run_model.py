import pandas as pd
import numpy as np

from typing import Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from parse_personnel import add_personnel_features
from add_participation_features import add_participation_features


def train_run_models() -> Dict[str, Dict[str, Any]]:
    """
    2-stage model pipeline:
    1. 'situation' model predicts run success probability.
    2. 'tendency' models use that probability as a feature
       to predict 'run_gap' and 'run_location'.

    This version keeps the original group structure, but
    adds optional personnel + formation features.
    """

    # 1. data loading
    try:
        pbp_files: List[pd.DataFrame] = [
            pd.read_csv("Data/pbp_2024_0.csv", low_memory=False),
            pd.read_csv("Data/pbp_2024_1.csv", low_memory=False),
        ]
        df: pd.DataFrame = pd.concat(pbp_files, ignore_index=True)
    except FileNotFoundError:
        print("Error: Data files not found in 'Data/' directory. Exiting.")
        return {}

    df_filtered: pd.DataFrame = df[df["play_type"] == "run"].copy()

    if df_filtered.empty:
        print("No 'run' plays found. Exiting.")
        return {}

    # 1b. OPTIONAL participation / personnel merge
    try:
        part_df: pd.DataFrame = pd.read_csv(
            "Data/pbp_participation_2024.csv",
            low_memory=False,
        )
    except FileNotFoundError:
        print("Warning: participation file not found, skipping personnel/formation features.")
        part_df = None

    if part_df is not None:
        # figure out shared game key
        if "old_game_id" in df_filtered.columns and "old_game_id" in part_df.columns:
            game_key_col: str = "old_game_id"
        elif "game_id" in df_filtered.columns and "game_id" in part_df.columns:
            game_key_col = "game_id"
        else:
            print("Warning: no shared game id between PBP and participation; skipping personnel/formation.")
            part_df = None

    if part_df is not None:
        keep_cols: List[str] = [game_key_col, "play_id"]

        # only grab what we actually care about right now
        extra_part_cols: List[str] = [
            "offense_personnel",
            "defense_personnel",
            "offense_formation",
            "defenders_in_box",
        ]

        for col in extra_part_cols:
            if col in part_df.columns:
                keep_cols.append(col)

        part_df = part_df[keep_cols].drop_duplicates(subset=[game_key_col, "play_id"])

        df_filtered = pd.merge(
            df_filtered,
            part_df,
            on=[game_key_col, "play_id"],
            how="left",
        )

        # rename into what add_personnel_features expects
        df_filtered = df_filtered.rename(
            columns={
                "offense_personnel": "personnel_off",
                "defense_personnel": "personnel_def",
            }
        )

        # engineer numeric personnel and helper flags
        df_filtered = add_personnel_features(df_filtered)
        df_filtered = add_participation_features(df_filtered)

    # 2. pre-processing (same as group version)
    for col in ["temp", "wind"]:
        if col in df_filtered.columns:
            median_val: float = float(df_filtered[col].median())
            df_filtered[col] = df_filtered[col].fillna(median_val)

    # new situational features
    required_cols: List[str] = [
        "yardline_100",
        "goal_to_go",
        "ydstogo",
        "down",
        "quarter_seconds_remaining",
        "qtr",
        "score_differential",
    ]

    if all(col in df_filtered.columns for col in required_cols):
        df_filtered["is_redzone"] = (df_filtered["yardline_100"] <= 20).astype(int)
        df_filtered["is_goal_line"] = (
            (df_filtered["goal_to_go"] == 1) & (df_filtered["yardline_100"] <= 10)
        ).astype(int)
        df_filtered["is_short_yardage"] = (
            (df_filtered["ydstogo"] <= 2) & (df_filtered["down"] >= 3)
        ).astype(int)
        df_filtered["is_two_minute_drill"] = (
            (df_filtered["quarter_seconds_remaining"] <= 120)
            & (df_filtered["qtr"].isin([2, 4]))
        ).astype(int)
        df_filtered["is_close_game_late"] = (
            (df_filtered["qtr"] == 4)
            & (df_filtered["score_differential"].abs() <= 8)
        ).astype(int)

    # 3. define 'y' (target) for Stage 1: 'run_success'
    if not all(col in df_filtered.columns for col in ["yards_gained", "ydstogo", "touchdown"]):
        print(
            "Error: Missing 'yards_gained', 'ydstogo', or 'touchdown'. "
            "Cannot create 'run_success' target. Exiting."
        )
        return {}

    is_first_down: pd.Series = (
        (df_filtered["yards_gained"] >= df_filtered["ydstogo"])
        & (df_filtered["yards_gained"].notna())
    )
    is_touchdown: pd.Series = df_filtered["touchdown"] == 1
    y_sit: pd.Series = (is_first_down | is_touchdown).astype(int)
    y_sit.name = "run_success"

    # 4. define 'X' (features) for all models

    # original base feature set from your group
    base_feature_columns: List[str] = [
        "down",
        "ydstogo",
        "yardline_100",
        "goal_to_go",
        "qtr",
        "quarter_seconds_remaining",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "score_differential",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "posteam",
        "defteam", 
        "shotgun",
        "no_huddle",
        "roof",
        "surface",
        "temp",
        "wind",
        "is_redzone",
        "is_goal_line",
        "is_short_yardage",
        "is_two_minute_drill",
        "is_close_game_late",
    ]

    # NEW: personnel numeric features (if created)
    personnel_numeric: List[str] = [
        "off_rb",
        "off_te",
        "off_wr",
        "def_dl",
        "def_lb",
        "def_db",
        "defenders_in_box",
    ]

    # NEW: personnel / formation categoricals
    personnel_categorical: List[str] = [
        "offense_formation",
        "off_group_bucket",
        "def_group_bucket",
    ]

    feature_columns: List[str] = (
        base_feature_columns + personnel_numeric + personnel_categorical
    )

    # fill numeric NaNs for personnel if present
    for num_col in personnel_numeric:
        if num_col in df_filtered.columns:
            df_filtered[num_col] = df_filtered[num_col].fillna(0)

    # #TODO: Temporarily removing the the personnel features and keeping only the base features
    # feature_columns: List[str] = (base_feature_columns)

    existing_features: List[str] = [
        col for col in feature_columns if col in df_filtered.columns
    ]
    X: pd.DataFrame = df_filtered[existing_features].copy()

    if X.empty:
        print("Error: No features found. Exiting.")
        return {}

    # encode categorical features
    categorical_cols: List[str] = [
        "posteam",
        "defteam",
        "roof",
        "surface",
        "qtr",
        "offense_formation",
        "off_group_bucket",
        "def_group_bucket",
    ]

    existing_categorical: List[str] = [
        col for col in categorical_cols if col in X.columns
    ]
    X_processed: pd.DataFrame = pd.get_dummies(
        X,
        columns=existing_categorical,
        drop_first=True,
    )

    # 5. Stage 1 model: predict run_success
    print("Starting stage 1 model training (Run Success)")

    valid_indices = y_sit.dropna().index
    X_sit: pd.DataFrame = X_processed.loc[valid_indices]
    y_sit = y_sit.loc[valid_indices]

    if y_sit.nunique() < 2:
        print("Error: 'run_success' target has only 1 class, cant train.")
        return {}

    X_sit_train, X_sit_test, y_sit_train, y_sit_test = train_test_split(
        X_sit,
        y_sit,
        test_size=0.2,
        random_state=42,
        stratify=y_sit,
    )

    sit_model: RandomForestClassifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    sit_model.fit(X_sit_train, y_sit_train)

    importances: np.ndarray = sit_model.feature_importances_
    feat_names: np.ndarray = sit_model.feature_names_in_

    sorted_feats: List[tuple[str, float]] = sorted(
        zip(feat_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\nAll situation features (by RF importance):")
    for name, score in sorted_feats:
        print(f"{name:<40} {score:.4f}")

    y_sit_pred: np.ndarray = sit_model.predict(X_sit_test)
    accuracy: float = accuracy_score(y_sit_test, y_sit_pred)
    print(f"\nStage 1 model ('run_success') Accuracy: {accuracy:.3f}")

    # 6. Stage 2 models: predict tendencies

    # generate new probability feature for both train and test sets
    X_tend_train: pd.DataFrame = X_sit_train.copy()
    X_tend_train["predicted_run_success_prob"] = sit_model.predict_proba(
        X_sit_train[sit_model.feature_names_in_]
    )[:, 1]

    X_tend_test: pd.DataFrame = X_sit_test.copy()
    X_tend_test["predicted_run_success_prob"] = sit_model.predict_proba(
        X_sit_test[sit_model.feature_names_in_]
    )[:, 1]

    trained_models: Dict[str, Dict[str, Any]] = {}
    target_columns: List[str] = ["run_gap", "run_location"]

    for target in target_columns:
        print(f"\nStarting stage 2 model training for: {target}")

        if target not in df_filtered.columns:
            print(f"Warning: Target column '{target}' not found.")
            continue

        y_tend: pd.Series = df_filtered[target]

        y_tend_train_valid: pd.Series = y_tend.loc[X_tend_train.index].dropna()
        X_tend_train_clean: pd.DataFrame = X_tend_train.loc[y_tend_train_valid.index]

        y_tend_test_valid: pd.Series = y_tend.loc[X_tend_test.index].dropna()
        X_tend_test_clean: pd.DataFrame = X_tend_test.loc[y_tend_test_valid.index]

        if X_tend_train_clean.empty or y_tend_train_valid.nunique() < 2:
            print(
                f"Warning: Not enough data or classes for '{target}' after cleaning. Skipping..."
            )
            continue

        model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
        model.fit(X_tend_train_clean, y_tend_train_valid)

        if not X_tend_test_clean.empty:
            y_pred: np.ndarray = model.predict(X_tend_test_clean)
            tend_accuracy: float = accuracy_score(y_tend_test_valid, y_pred)

            print(f"Stage 2 Model accuracy for '{target}': {tend_accuracy:.3f}")
            print(f"Classification report for '{target}':")
            labels: np.ndarray = np.union1d(y_tend_test_valid.unique(), y_pred)
            print(
                classification_report(
                    y_tend_test_valid,
                    y_pred,
                    labels=labels,
                    zero_division=0,
                )
            )

        print(f"Model training complete for: {target}")

        trained_models[target] = {
            "model": model,
            "columns": X_tend_train_clean.columns.tolist(),
        }

    # store situation model too
    trained_models["situation"] = {
        "model": sit_model,
        "columns": X_sit_train.columns.tolist(),
    }

    return trained_models


def predict_run_metrics(situation, trained_models):
    ''' 
        Function that predicts the most optimal run gap and location 
        with success probability for this specific run play 
        based on the trained run models. 
    '''

    # Convert input to DataFrame with correct column names
    situation_df = pd.DataFrame([situation], columns=['down', 'ydstogo', 'yardline_100', 'goal_to_go', 'quarter_seconds_remaining',
                                                      'half_seconds_remaining', 'game_seconds_remaining', 'score_differential', 
                                                      'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'posteam', 'defteam'])
    
    # Add the engineered features (same as in training)
    situation_df["is_redzone"] = (situation_df["yardline_100"] <= 20).astype(int)
    situation_df["is_goal_line"] = (
        (situation_df["goal_to_go"] == 1) & (situation_df["yardline_100"] <= 10)
    ).astype(int)
    situation_df["is_short_yardage"] = (
        (situation_df["ydstogo"] <= 2) & (situation_df["down"] >= 3)
    ).astype(int)
    
    # Infer quarter from time remaining (simple heuristic)
    if situation[6] > 2700:  # game_seconds_remaining
        qtr = 1
    elif situation[6] > 1800:
        qtr = 2
    elif situation[6] > 900:
        qtr = 3
    else:
        qtr = 4
    
    situation_df["qtr"] = qtr
    situation_df["is_two_minute_drill"] = (
        (situation_df["quarter_seconds_remaining"] <= 120)
        & (situation_df["qtr"].isin([2, 4]))
    ).astype(int)
    situation_df["is_close_game_late"] = (
        (situation_df["qtr"] == 4)
        & (situation_df["score_differential"].abs() <= 8)
    ).astype(int)
    
    # Add default values for missing columns that were used in training
    situation_df["shotgun"] = 0  # default value
    situation_df["no_huddle"] = 0  # default value
    situation_df["roof"] = "outdoors"  # default value
    situation_df["surface"] = "grass"  # default value
    situation_df["temp"] = 70  # default value
    situation_df["wind"] = 0  # default value
    
    # Apply one-hot encoding to categorical columns (same as training)
    categorical_cols = ["posteam", "defteam", "roof", "surface", "qtr"]
    situation_encoded = pd.get_dummies(
        situation_df,
        columns=categorical_cols,
        drop_first=True,
    )
    
    # Get the situation model to predict success probability first
    sit_model_info = trained_models.get("situation")
    if sit_model_info:
        sit_model = sit_model_info["model"]
        sit_columns = sit_model_info["columns"]
        
        # Align columns with training data
        for col in sit_columns:
            if col not in situation_encoded.columns:
                situation_encoded[col] = 0
        
        situation_encoded = situation_encoded[sit_columns]
        
        # Predict success probability
        success_prob = sit_model.predict_proba(situation_encoded)[0, 1]
        print(f"Predicted run success probability: {success_prob:.3f}")
        
        # Add this as a feature for tendency models
        situation_encoded["predicted_run_success_prob"] = success_prob
    
    # Predict the most optimal metric (gap, location) for the run play 
    run_metrics = {}
    for metric in ["run_gap", "run_location"]:
        if metric in trained_models:
            model_info = trained_models[metric]
            model = model_info["model"]
            model_columns = model_info["columns"]
            
            # Align columns with training data
            for col in model_columns:
                if col not in situation_encoded.columns:
                    situation_encoded[col] = 0
            
            # Reorder columns to match training
            situation_for_prediction = situation_encoded[model_columns]
            
            # Predict
            prediction = model.predict(situation_for_prediction)[0]
            run_metrics[metric] = prediction
            print(f"Predicted {metric}: {prediction}")

    return run_metrics



if __name__ == "__main__":
    print("Starting all run model training")
    all_models: Dict[str, Dict[str, Any]] = train_run_models()

    if all_models:
        print("\nAll model training complete")
        print(f"Trained {len(all_models)} models: {list(all_models.keys())}")
    else:
        print("\nModel training failed ---")
