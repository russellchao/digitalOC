import pandas as pd
import numpy as np
import nflreadpy as nfl

pbp = pd.read_csv("data/pbp_2024_0.csv", low_memory=False)
pbp_subset = pbp[
    [
        "old_game_id", "play_id",       
        "posteam", "defteam",           
        "play_type", "pass_attempt", "rush_attempt", "air_yards", "run_gap",  
        "qb_kneel", "qb_spike", "qb_scramble", "sack",  
        "down", "yards_gained", "ydstogo",  
        "epa"  
    ]
]

part = pd.read_csv(
    "data/pbp_participation_2024.csv",
    usecols=[
        "old_game_id", "play_id",
        "was_pressure", "number_of_pass_rushers",
        "offense_personnel", "offense_formation", "defense_personnel"
    ]
)

merged = pd.merge(part, pbp_subset, on=["old_game_id", "play_id"], how="left")


class PlayClassifier:
    @staticmethod
    def get_category(row):
        # ignore non‑offensive plays
        if row.get("play_type") not in ["pass", "run"]:
            return "other"

        # special QB situations
        if row.get("qb_kneel") == 1:
            return "qb_kneel"
        elif row.get("qb_spike") == 1:
            return "qb_spike"
        elif row.get("qb_scramble") == 1:
            return "qb_scramble"
        elif row.get("sack") == 1:
            return "sack"

     
        if row.get("pass_attempt") == 1:
            air_yards = row.get("air_yards")
            if pd.isna(air_yards):
                return "pass"
            try:
                air_yards = float(air_yards)
                if air_yards < 0:
                    return "screen_pass"
                elif air_yards <= 7:
                    return "short_pass"
                elif air_yards <= 15:
                    return "mid_pass"
                else:
                    return "deep_pass"
            except:
                return "pass"
            


        # --- RUN PLAYS ---
        elif row.get("rush_attempt") == 1:
            gap = str(row.get("run_gap")).lower() if pd.notna(row.get("run_gap")) else ""
            if gap in ["guard", "center"]:
                return "inside_run"
            elif gap == "end":
                return "outside_run"


        return "other"


        
class Team:
    def __init__(self, name: str, df: pd.DataFrame):

        self.name = name
        self.df = df

    def offensive_snaps(self):
        return self.df[self.df["posteam"] == self.name]["play_id"].nunique()

    def offensive_personnel_counts(self):
        return self.df[self.df["posteam"] == self.name]["offense_personnel"].value_counts()

    def formation_counts(self):
        return self.df[self.df["posteam"] == self.name]["offense_formation"].value_counts()

    def defensive_personnel_faced(self):
        return self.df[self.df["posteam"] == self.name]["defense_personnel"].value_counts()

    def pressure_rate(self):
        return self.df[self.df["posteam"] == self.name]["was_pressure"].mean()
    
    @staticmethod
    def success_rule(row):
        """Return True if play is successful based on down and yards gained."""
        if row["down"] == 1:
            return row["yards_gained"] >= 0.5 * row["ydstogo"]
        elif row["down"] == 2:
            return row["yards_gained"] >= 0.7 * row["ydstogo"]
        elif row["down"] in [3, 4]:
            return row["yards_gained"] >= row["ydstogo"]
        return False

    
    def success_rate_by_down(self):
        df = self.df[self.df["posteam"] == self.name].copy()
        df = df.dropna(subset=["down", "yards_gained", "ydstogo"])

        df["successful"] = df.apply(self.success_rule, axis=1)

        rates = (
            df.groupby("down")["successful"]
            .mean()
            .round(3)
            .rename("success_rate")
        )

        return rates
    
    def success_rate_by_down(self):
        df = self.df[self.df["posteam"] == self.name].copy()
        df = df.dropna(subset=["down", "yards_gained", "ydstogo"])

        df["successful"] = df.apply(self.success_rule, axis=1)

        rates = (
            df.groupby("down")["successful"]
            .mean()
            .round(3)
            .rename("success_rate")
        )

        return rates
    
    def success_rate_by_playType(self):
        df = self.df[self.df["posteam"] == self.name].copy()
        df = df.dropna(subset=["down", "yards_gained", "ydstogo"])

        df["successful"] = df.apply(self.success_rule, axis=1)

        rates = (
            df.groupby("play_type")["successful"]
            .mean()
            .round(3)
            .rename("success_rate")
        )

        return rates
    
    def playType_by_down(self):
        # Filter to this team's offensive plays
        df = self.df[self.df["posteam"] == self.name].copy()
        
        # Keep only rows with valid down and play_type
        df = df.dropna(subset=["down", "play_type"])
        
        # Count plays by down and play_type
        counts = df.groupby(["down", "play_type"]).size().unstack(fill_value=0)
        
        # Convert counts to percentages
        fractions = counts.div(counts.sum(axis=1), axis=0).round(3)
        
        return fractions
    
    def blitz_rate_by_down(self):
        # Filter for plays where this team was on defense
        df = self.df[self.df["defteam"] == self.name].copy()

        # Keep only rows with valid down and number_of_pass_rushers
        df = df.dropna(subset=["down", "number_of_pass_rushers"])

        # Define a blitz
        df["blitz"] = df["number_of_pass_rushers"] > 4

        # Calculate blitz rate by down
        blitz_rates = (
            df.groupby("down")["blitz"]
            .mean()
            .round(3)
            .rename("blitz_rate")
        )

        return blitz_rates
    
    def success_against_blitz(self):
        
        df = self.df[self.df["posteam"] == self.name].copy()

        
        df = df.dropna(subset=["number_of_pass_rushers", "yards_gained", "was_pressure"])

        
        blitz_df = df[df["number_of_pass_rushers"] > 4]

        if blitz_df.empty:
            return 0.0  

       
        blitz_df["unsuccessful"] = (blitz_df["was_pressure"]) | (blitz_df["yards_gained"] < 0)

       
        success_ratio = 1 - blitz_df["unsuccessful"].mean()

        return round(success_ratio, 3)

    def offensive_category_stats(self):
        df = self.df[self.df["posteam"] == self.name].copy()
        df = df.dropna(subset=["play_category", "yards_gained", "down", "ydstogo"])
        df["successful"] = df.apply(self.success_rule, axis=1)

        stats = (
            df.groupby("play_category")
            .agg(
                num_plays=("play_id", "count"),
                success_rate=("successful", "mean"),
                avg_yards=("yards_gained", "mean"),
                avg_epa=("epa", "mean")
            )
            .sort_values("num_plays", ascending=False)
            .round(3)
        )

        return stats

    


    

    def __repr__(self):
        return f"<Team {self.name}: {self.offensive_snaps()} offensive snaps>"

merged["play_category"] = merged.apply(PlayClassifier.get_category, axis=1)
def compute_elo_per_play_type(category_stats: pd.DataFrame) -> dict:
    """
    Compute a separate ELO-style score for each primary play category.
    Returns a dictionary: {category_name: elo_score}
    """
    # Define weights
    w_success = 0.4
    w_epa = 0.6

    elo_scores = {}

    # Only include relevant play types
    for category in ["deep_pass", "mid_pass", "short_pass", "screen_pass", "inside_run", "outside_run"]:
        if category in category_stats.index:
            row = category_stats.loc[category]
            success = row.get("success_rate", 0)
            epa = row.get("avg_epa", 0)

            # Normalize metrics
            norm_success = (success - 0.4) / (0.6 - 0.4)  # Assume success ~ 0.4 to 0.6
            norm_epa = (epa + 0.5) / 1.0                  # Assume EPA ~ -0.5 to +0.5

            
            norm_success = np.clip(norm_success, 0, 1)
            norm_epa = np.clip(norm_epa, 0, 1)

          
            elo = 1000 + 400 * (w_success * norm_success + w_epa * norm_epa)
            elo_scores[category] = round(elo, 2)
        else:
            elo_scores[category] = 1000.0  

    return elo_scores

if __name__ == "__main__":
    kc = Team("LV", merged)
    category_stats = kc.offensive_category_stats()
    per_type_elo = compute_elo_per_play_type(category_stats)

    print(f"ELO Ratings by Play Type for {kc.name}:")
    for play_type, rating in per_type_elo.items():
        print(f"{play_type:<15}: {rating}")
