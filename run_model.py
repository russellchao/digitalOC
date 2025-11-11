import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def train_run_models():
    """
    2-stage model pipeline:
    1. 'situation' model predicts run success probability.
    2. 'tendency' models use that probability as a feature
       to predict 'run_gap' and 'run_location'.
    """
    # 1. data loading 
    try:
        pbp_files = [pd.read_csv("Data/pbp_2024_0.csv", low_memory=False), pd.read_csv("Data/pbp_2024_1.csv", low_memory=False)]
        df = pd.concat(pbp_files, ignore_index=True)
    except FileNotFoundError:
        print("Error: Data files not found in 'Data/' directory. Exiting.")
        return None

    df_filtered = df[df['play_type'] == 'run'].copy()
    
    if df_filtered.empty:
        print("No 'run' plays found. Exiting.")
        return None

    # 2. pre processing
    for col in ['temp', 'wind']:
        if col in df_filtered.columns:
            median_val = df_filtered[col].median()
            df_filtered[col] = df_filtered[col].fillna(median_val)

    # new situational features
    required_cols = ['yardline_100', 'goal_to_go', 'ydstogo', 'down', 
                     'quarter_seconds_remaining', 'qtr', 'score_differential']
    
    if all(col in df_filtered.columns for col in required_cols):
        df_filtered['is_redzone'] = (df_filtered['yardline_100'] <= 20).astype(int)
        df_filtered['is_goal_line'] = ((df_filtered['goal_to_go'] == 1) & (df_filtered['yardline_100'] <= 10)).astype(int)
        df_filtered['is_short_yardage'] = ((df_filtered['ydstogo'] <= 2) & (df_filtered['down'] >= 3)).astype(int)
        df_filtered['is_two_minute_drill'] = ((df_filtered['quarter_seconds_remaining'] <= 120) & (df_filtered['qtr'].isin([2, 4]))).astype(int)
        df_filtered['is_close_game_late'] = ((df_filtered['qtr'] == 4) & (df_filtered['score_differential'].abs() <= 8)).astype(int)
    
    # define 'y' (target) for Stage 1: 'run_success'
    if not all(col in df_filtered.columns for col in ['yards_gained', 'ydstogo', 'touchdown']):
        print("Error: Missing 'yards_gained', 'ydstogo', or 'touchdown'. Cannot create 'run_success' target. Exiting.")
        return None
        
    is_first_down = (df_filtered['yards_gained'] >= df_filtered['ydstogo']) & (df_filtered['yards_gained'].notna())
    is_touchdown = (df_filtered['touchdown'] == 1)
    y_sit = (is_first_down | is_touchdown).astype(int)
    y_sit.name = "run_success"

    # define 'X' (features) for all models
    feature_columns = [
        'down', 'ydstogo', 'yardline_100', 'goal_to_go', 'qtr',
        'quarter_seconds_remaining', 'game_seconds_remaining', 
        'score_differential', 'posteam_timeouts_remaining', 
        'defteam_timeouts_remaining', 'shotgun', 'no_huddle',
        'posteam', 'defteam', 'roof', 'surface', 'temp', 'wind',
        'is_redzone', 'is_goal_line', 'is_short_yardage', 
        'is_two_minute_drill', 'is_close_game_late'
    ]
    
    existing_features = [col for col in feature_columns if col in df_filtered.columns]
    X = df_filtered[existing_features].copy()
    
    if X.empty:
        print("Error: No features found. Exiting.")
        return None
        
    # encode all categorical features once
    categorical_cols = ['posteam', 'defteam', 'roof', 'surface', 'qtr']
    existing_categorical = [col for col in categorical_cols if col in X.columns]
    X_processed = pd.get_dummies(X, columns=existing_categorical, drop_first=True)

    # 3. stage 1 model: predict run success
    print("Starting stage 1 model training (Run Success)")

    valid_indices = y_sit.dropna().index
    X_sit = X_processed.loc[valid_indices]
    y_sit = y_sit.loc[valid_indices]

    if y_sit.nunique() < 2:
        print("Error: 'run_success' target has only 1 class, cant train.")
        return None

    X_sit_train, X_sit_test, y_sit_train, y_sit_test = train_test_split(
        X_sit, y_sit, test_size=0.2, random_state=42, stratify=y_sit
    )

    sit_model = RandomForestClassifier(n_estimators=100, random_state=42)
    sit_model.fit(X_sit_train, y_sit_train)
    
    y_sit_pred = sit_model.predict(X_sit_test)
    accuracy = accuracy_score(y_sit_test, y_sit_pred)
    print(f"Stage 1 model ('run_success') Accuracy: {accuracy:.3f}")

    # 4. stage 2 models: predict tendencies
    
    # generate new probability feature for both train and test sets
    X_tend_train = X_sit_train.copy()
    X_tend_train['predicted_run_success_prob'] = sit_model.predict_proba(X_sit_train[sit_model.feature_names_in_])[:, 1]
    
    X_tend_test = X_sit_test.copy()
    X_tend_test['predicted_run_success_prob'] = sit_model.predict_proba(X_sit_test[sit_model.feature_names_in_])[:, 1]
    
    trained_models = {}
    target_columns = ['run_gap', 'run_location'] 

    for target in target_columns:
        print(f"\nStarting stage 2 model training for: {target}")

        if target not in df_filtered.columns:
            print(f"Warning: Target column '{target}' not found.")
            continue
            
        y_tend = df_filtered[target]
        
        y_tend_train_valid = y_tend.loc[X_tend_train.index].dropna()
        X_tend_train_clean = X_tend_train.loc[y_tend_train_valid.index]
        
        y_tend_test_valid = y_tend.loc[X_tend_test.index].dropna()
        X_tend_test_clean = X_tend_test.loc[y_tend_test_valid.index]

        if X_tend_train_clean.empty or y_tend_train_valid.nunique() < 2:
            print(f"Warning: Not enough data or classes for '{target}' after cleaning. Skipping...")
            continue

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_tend_train_clean, y_tend_train_valid)
        
        if not X_tend_test_clean.empty:
            y_pred = model.predict(X_tend_test_clean)
            accuracy = accuracy_score(y_tend_test_valid, y_pred)
            
            print(f"Stage 2 Model accuracy for '{target}': {accuracy:.3f}")
            print(f"Classification report for '{target}':")
            labels = np.union1d(y_tend_test_valid.unique(), y_pred)
            print(classification_report(y_tend_test_valid, y_pred, labels=labels, zero_division=0))
        
        print(f"model training complete for: {target}")
        
        trained_models[target] = {
            'model': model,
            'columns': X_tend_train_clean.columns.tolist()
        }

    trained_models['situation'] = {
        'model': sit_model,
        'columns': X_sit_train.columns.tolist()
    }
    
    return trained_models




# --- This goes at the very end of your file ---
if __name__ == "__main__":
    print("Starting all run model training")
    all_models = train_run_models()

    if all_models:
        print(f"\nAll model training complete")
        print(f"Trained {len(all_models)} models: {list(all_models.keys())}")
    else:
        print("\nmodel training failed ---")