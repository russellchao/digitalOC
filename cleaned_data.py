import pandas as pd

# Open the CSV file
df = pd.read_csv("data/pbp_2024_0.csv", low_memory=False)

'''
Inputs for model:
Down (down)
Distance (ydstogo)
Time (quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining)
Quarter (qtr)
Yard line (yardline_100)
Difference of score between posteam and defteam (score_differential) -> (- if losing, + if winning)
Possessing team (posteam)
Defending team (defteam)
Timeouts remaining (posteam_timeouts_remaining)

Run Outcome / Run Called
Yards gained
Touchdown? (rush_touchdown) (1, 0)
First down (first_down_run) (1, 0)
Formation
Run play? (rush_attempt) (1 if run, 0 if not)
Type of run play (dive, sweep, etc)
run_location (left, middle, right)
run_gap (end, guard, tackle)
tackled_for_loss (1,0)
wpa	Win probability added (WPA) for the posteam.
epa	Expected points added (EPA) by the posteam for the given play.
'''


# Select relevant columns for analysis
relevant_columns =  [
        "old_game_id", "play_id",
        "posteam", "defteam",
        "play_type", "pass_attempt", "rush_attempt", "air_yards", "run_gap",
        "qb_kneel", "qb_spike", "qb_scramble", "sack",
        "down", "yards_gained", "ydstogo",
        "epa"
    ]

# Pull relevant data
# Clean any rows with 'NaN' data
df_relevant = df[relevant_columns]
df_cleaned = df_relevant.dropna()

# Save your DataFrame to a new CSV file
df_cleaned.to_csv("data/2024_0_EloData.csv", index=False)

# Compare dataframe sizes after cleaning
print("Size of dataframe before cleaning: ", df_relevant.shape)
print("Size of dataframe after cleaning: ", df_cleaned.shape)

# Print the first 50 rows of the cleaned data
print(df_cleaned.head(10))

# print(type(df))
# print(type(df_relevant))
# print(type(df_cleaned))

