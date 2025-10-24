# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_model(X, y):
    # Split the data between X and y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle categorical columns for the X values (non-numeric data)
    # (E.g. posteam="KC" becomes posteam_KC=1, all other team columns = 0)
    categorical_cols = ['posteam', 'defteam']
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True) # Using pd.get_dummies (one-hot encoding)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0) # Make sure train and test have same columns (important!)

    print(f"Training data shape before cleaning: {X_train_encoded.shape}")
    print(f"Test data shape before cleaning: {X_test_encoded.shape}")
    print()

    # Drop rows with missing values 
    train_complete_idx = X_train_encoded.dropna().index.intersection(y_train.dropna().index)
    X_train_clean = X_train_encoded.loc[train_complete_idx]
    y_train_clean = y_train.loc[train_complete_idx]
    test_complete_idx = X_test_encoded.dropna().index.intersection(y_test.dropna().index) # Do the same for test data
    X_test_clean = X_test_encoded.loc[test_complete_idx]
    y_test_clean = y_test.loc[test_complete_idx]

    print(f"Training data shape after cleaning: {X_train_clean.shape}")
    print(f"Test data shape after cleaning: {X_test_clean.shape}")
    print()

    # Create and train the model 
    model = RandomForestClassifier(n_estimators=100, random_state=42) # Initialize the classifier
    model.fit(X_train_clean, y_train_clean) # Train the model
    y_pred = model.predict(X_test_clean) # Make predictions on test set
    accuracy = accuracy_score(y_test_clean, y_pred) # Calculate accuracy
    
    # Print accuracy
    print(f"Model accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test_clean, y_pred))
    print()
    
    # Return the trained model for later use
    return model, X_train_clean.columns.tolist()  # Also return column names for later predictions


def predict_play(situation, trained_model, feature_columns):
    ''' Use the situation to determine the most optimal play type '''

    # Print the current situation
    print(f"Down: {situation[0]}")
    print(f"Yards to go: {situation[1]}")
    print(f"Distance to end zone: {situation[2]}")
    print(f"Goal to go: {situation[3]}")
    print(f"Quarter seconds remaining: {situation[4]}")
    print(f"Half seconds remaining: {situation[5]}")
    print(f"Game seconds remaining: {situation[6]}")
    print(f"Score differential: {situation[7]}")
    print(f"Win probability: {situation[8] * 100:.2f}%")
    print(f"Expected points: {situation[9]}")
    print(f"Offensive team timeouts remaining: {situation[10]}")
    print(f"Defensive team timeouts remaining: {situation[11]}")
    print(f"Offensive team: {situation[12]}")
    print(f"Defensive team: {situation[13]}")
    print()

    # Convert input to DataFrame with correct column names
    situation_df = pd.DataFrame([situation], columns=['down', 'ydstogo', 'yardline_100', 'goal_to_go', 'quarter_seconds_remaining', 
                                                      'half_seconds_remaining', 'game_seconds_remaining', 'score_differential', 'wp', 
                                                      'ep', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'posteam', 'defteam'])

    # Apply same categorical encoding as training
    categorical_cols = ['posteam', 'defteam']
    situation_encoded = pd.get_dummies(situation_df, columns=categorical_cols, drop_first=True)
    
    # Make sure it has all the same columns as training data
    for col in feature_columns:
        if col not in situation_encoded.columns:
            situation_encoded[col] = 0
    
    # Reorder columns to match training data
    situation_encoded = situation_encoded[feature_columns]

    # Predict the most optimal play type
    prediction = trained_model.predict(situation_encoded)
    prediction_proba = trained_model.predict_proba(situation_encoded)

    print("======================================")
    print(f"Predicted Play Type: {prediction}")
    print(f"Confidence {prediction_proba}")
    print("======================================")
    print()
    
    return prediction[0], prediction_proba[0]  # Return prediction and confidence








if __name__ == "__main__":
    ''' For testing only. Will eventually add a function with the same functionality below that will be called from an endpoint ''' 

    # Open both 2024 Play-by-Play CSV files and combine them
    pbp_files = [pd.read_csv("Data/pbp_2024_0.csv"), pd.read_csv("Data/pbp_2024_1.csv")]
    df = pd.concat(pbp_files, ignore_index=True)
    print(df.columns.to_list())
    print(df.head())
    
    # Filter columns that only contain "run" or "pass" for play_type
    df_filtered = df[df['play_type'].isin(['run', 'pass'])]
    print(f"Number of rows after filtering for run/pass plays: {df_filtered.shape[0]}")

    # (game situation) x variables using categories 1-3 for now
    X = df_filtered[['down', 'ydstogo', 'yardline_100', 'goal_to_go', 'quarter_seconds_remaining',
            'half_seconds_remaining', 'game_seconds_remaining', 'score_differential', 'wp',
            'ep', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'posteam', 'defteam']]
    print(X.head(10))

    # y variables
    y = df_filtered['play_type']
    print(y.head(10))

    trained_model, feature_columns = train_model(X, y)
    print(feature_columns)
    print()

    # Test situations to predict play type - [down, ydstogo, yardline_100, goal_to_go, quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining, score_differential, wp, ep, posteam_timeouts_remaining, defteam_timeouts_remaining, posteam, defteam]
    test_case_1 = [2, 5, 30, 0, 720, 720, 2520, 0, 0.52, 1.8, 3, 3, 'KC', 'BUF'] # 2nd & 5 from opponent's 30-yard line, Q2-12:00, tied game, balanced situation
    test_case_2 = [3, 8, 50, 0, 180, 1080, 1080, -3, 0.42, 0.8, 2, 3, 'GB', 'DAL'] # 3rd & 8 from midfield, Q3-3:00, down by 3, passing situation
    test_case_3 = [1, 10, 75, 0, 480, 1380, 3180, 7, 0.62, -0.4, 3, 3, 'SF', 'SEA'] # 1st & 10 from own 25-yard line, Q1-8:00, ahead by 7, balanced situation
    test_case_4 = [1, 8, 8, 1, 95, 95, 95, -4, 0.15, 4.2, 1, 2, 'NE', 'NYG'] # 1st & Goal from 8-yard line, Q4-1:35, down by 4, red zone situation  
    test_case_5 = [4, 2, 35, 0, 45, 45, 45, -6, 0.05, 1.2, 0, 1, 'TB', 'DET'] # 4th & 2 from opponent's 35, Q4-0:45, down by 6, desperation situation
    test_case_6 = [3, 1, 60, 0, 600, 1500, 3300, 0, 0.50, 0.5, 2, 2, 'MIA', 'NYJ'] # 3rd & 1 from own 40, Q1-10:00, tied game, short yardage situation
    test_case_7 = [1, 1, 1, 1, 600, 600, 2400, 0, 0.50, 0.5, 2, 2, 'PHI', 'LAR'] # 1st & Goal from the 1, Q3-10:00, Tush Push Situation for the Eagles 
    test_case_8 = [3, 15, 80, 0, 900, 900, 900, -10, 0.30, -1.5, 1, 2, 'CIN', 'PIT'] # 3rd and long from own 20, Q4-15:00, down by 10, passing situation

    all_test_cases = [test_case_1, test_case_2, test_case_3, test_case_4, test_case_5, test_case_6, test_case_7, test_case_8]

    for test_case in all_test_cases:
        predict_play(test_case, trained_model, feature_columns)
    