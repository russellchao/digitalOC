import pandas as pd

# Load raw data
df = pd.read_csv("digitalOC/data/pbp_2020_0.csv", low_memory=False)

useful_cols =[
    #Game context
    "game_id", "season_type", "week", "game_date",
    "home_team", "away_team", "posteam", "defteam", "location",
    "stadium", "roof", "surface", "temp", "wind",

    #Situation
    "qtr", "down", "ydstogo", "yardline_100",
    "time", "quarter_seconds_remaining", "game_seconds_remaining",
    "score_differential", "goal_to_go",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",

    #Play details
    "play_type", "rush_attempt", "pass_attempt", "field_goal_attempt",
    "shotgun", "no_huddle", "qb_dropback",
    "run_location", "run_gap", "pass_length", "pass_location",
    "air_yards", "yards_after_catch",

    #Outcomes
    "yards_gained", "success", "epa", "touchdown"
]

# Keep only the columns that exist in your dataset
useful_cols = [col for col in useful_cols if col in df.columns]
df = df[useful_cols]
# Drop plays that aren’t meaningful decisions
df = df[df["play_type"].isin(["run", "pass", "field_goal", "punt"])]
# Drop rows missing critical info
df = df.dropna(subset=["down", "ydstogo", "yardline_100"])
# Save cleaned version
df.to_csv("digitalOC/data/pbp_2020_cleaned.csv", index=False)
print(" Cleaned data saved to digitalOC/data/pbp_2020_cleaned.csv")
