import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def train_pass_model():
    """
    Train a pass success prediction model using pre-snap features.
    """

    df = pd.read_csv("Data/pbp_2024_0.csv", low_memory=False)
    # only include pass plays
    df_pass = df[df['play_type'] == 'pass'].copy()

    is_first_down = (df_pass['yards_gained'] >= df_pass['ydstogo']) & (df_pass['yards_gained'].notna())
    is_touchdown = (df_pass['touchdown'] == 1)
    df_pass['pass_success'] = (is_first_down | is_touchdown).astype(int)
    df_pass.loc[:,'pass_success'] = (is_first_down | is_touchdown).astype(int)

    categorical_cols = ["posteam", "defteam"]
    df_pass = pd.get_dummies(df_pass, columns=categorical_cols, drop_first=True)
    df_pass[['shotgun', 'no_huddle', 'qb_dropback']] = df_pass[['shotgun', 'no_huddle', 'qb_dropback']].fillna(0)


    # filtered columns needed to train the pass model
    pre_snap_info = [
        # situation info is stuff that will be inputted to the model
        # pre snap offensive scheme and pass details is stuff that we want the model to predict/recommend
        # situation info
        'down', 'ydstogo', 'yardline_100', 'goal_to_go',
        'qtr', 'quarter_seconds_remaining', 'half_seconds_remaining', 'game_seconds_remaining',
        'score_differential', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
        'shotgun', 'no_huddle', 'qb_dropback',
    ]
    dummy_cols = [col for col in df_pass.columns if col.startswith('posteam_') or col.startswith('defteam_')]
    X = df_pass[pre_snap_info + (dummy_cols)]

    post_snap_info = [ # pass details
        'pass_length', 'pass_location', 'air_yards',

        # post play details
        'yards_after_catch', 'yards_gained', 'epa', 'success', 'wpa', 'complete_pass', 
        'air_epa' #includes hypothetical EPA from incompletions. (could be useful for good plays with drops or bad throws)

        # add route & personnel data from participation files
        # add our personalized success column
    ]

    X = df_pass[pre_snap_info]  # features available before the snap
    y = df_pass['pass_success'] # target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pass_success_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    pass_success_model.fit(X_train, y_train)

    y_pred = pass_success_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    df_pass['predicted_pass_success_prob'] = pass_success_model.predict_proba(X)[:, 1]
    return pass_success_model, X.columns.tolist(), df_pass
print(train_pass_model())
