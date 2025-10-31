import pandas as pd

df = pd.read_csv("Data/pbp_2024_0.csv")

# only include pass plays
df_pass = df[df['play_type'] == 'pass']

# filtered columns needed to train the pass model
filtered_cols = [

    # situation info is stuff that will be inputted to the model
    # pre snap offensive scheme and pass details is stuff that we want the model to predict/recommend

    # situation info
    'down', 'ydstogo', 'yardline_100', 'goal_to_go',
    'qtr', 'quarter_seconds_remaining', 'half_seconds_remaining', 'game_seconds_remaining',
    'score_differential', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
    'posteam', 'defteam',

    # pre snap offensive scheme
    'shotgun', 'no_huddle', 'qb_dropback',

    # pass details
    'pass_length', 'pass_location', 'air_yards',

    # post play details
    'yards_after_catch', 'yards_gained', 'epa', 'success', 'wpa', 'complete_pass', 
    'air_epa' #includes hypothetical EPA from incompletions. (could be useful for good plays with drops or bad throws)

    # add route & personnel data from participation files
    # add our personalized success column

]

df_pass_filtered = df_pass[filtered_cols].copy()


# using only parameters from the pre-cleaned csvs

# filtered_cols = [
# 'goal_to_go', 'no_huddle', 'posteam_timeouts_remaining', 'yards_after_catch', 
# 'quarter_seconds_remaining', 'qb_dropback', 'air_yards', 'down', 'defteam_timeouts_remaining', 
# 'success', 'pass_location', 'score_differential', 'posteam', 'shotgun', 'ydstogo', 'yards_gained', 
# 'yardline_100', 'qtr', 'defteam', 'epa', 'game_seconds_remaining', 'pass_length'
# ]









