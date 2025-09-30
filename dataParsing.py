import nflreadpy as nfl
import pandas as pd

# Load CSVs
part = pd.read_csv("data/pbp_participation_2024.csv")
pbp = pd.read_csv("data/pbp_2024.csv")

# Merge participation with play-by-play context
merged = pd.merge(
    part,
    pbp[
        [
            "old_game_id", "play_id",
            "posteam", "defteam",
            "posteam_score", "defteam_score", "score_differential",
            "posteam_timeouts_remaining", "defteam_timeouts_remaining"
        ]
    ],
    left_on=["old_game_id", "play_id"],   # participation file keys
    right_on=["old_game_id", "play_id"],  # pbp file keys
    how="left"
)


def get_offensive_snap_count(df, team):
    return df[df["posteam"] == team]["play_id"].nunique()

def get_personnel_counts(df, team):
    return df[df["posteam"] == team]["offense_personnel"].value_counts()

def get_formation_counts(df, team):
    return df[df["posteam"] == team]["offense_formation"].value_counts()

def get_defense_personnel_counts(df, team):
    return df[df["posteam"] == team]["defense_personnel"].value_counts()

def get_pressure_rate(df, team):
    return df[df["posteam"] == team]["was_pressure"].mean()

def get_possession_defense_teams(df, play_id):
    row = df[df["play_id"] == play_id]
    if row.empty:
        return None, None
    row = row.iloc[0]
    return row["posteam"], row["defteam"]

def get_score_differential(df, play_id):
    row = df[df["play_id"] == play_id]
    return None if row.empty else row.iloc[0]["score_differential"]

def get_timeouts(df, play_id):
    row = df[df["play_id"] == play_id]
    if row.empty:
        return None, None
    row = row.iloc[0]
    return row["posteam_timeouts_remaining"], row["defteam_timeouts_remaining"]


if __name__ == "__main__":
    print(pbp.columns.tolist())
    # Example usage on merged dataset
    print("KC offensive snaps:", get_offensive_snap_count(merged, "KC"))
    print("KC personnel counts:\n", get_personnel_counts(merged, "KC"))
    print("KC Formation Usage:\n", get_formation_counts(merged, "KC"))
    print("KC Defensive Personnel Faced:\n", get_defense_personnel_counts(merged, "KC"))
    print("KC pressure rate:", get_pressure_rate(merged, "KC"))
    print("Play teams (example):", get_possession_defense_teams(merged, 40))
    print("Score diff (example):", get_score_differential(merged, 40))
    print("Timeouts (example):", get_timeouts(merged, 40))
