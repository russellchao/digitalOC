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
            "posteam_timeouts_remaining", "defteam_timeouts_remaining",
            "down", "yards_gained", "ydstogo", "play_type"
        ]
    ],
    left_on=["old_game_id", "play_id"],
    right_on=["old_game_id", "play_id"],
    how="left"
)


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
        # Filter to this team's offensive plays
        df = self.df[self.df["posteam"] == self.name].copy()

        # Keep only plays with valid pass rusher and yard data
        df = df.dropna(subset=["number_of_pass_rushers", "yards_gained", "was_pressure"])

        # Filter only blitzed plays (more than 4 rushers)
        blitz_df = df[df["number_of_pass_rushers"] > 4]

        if blitz_df.empty:
            return 0.0  # avoid division by zero if no blitz data

        # Determine unsuccessful plays
        blitz_df["unsuccessful"] = (blitz_df["was_pressure"]) | (blitz_df["yards_gained"] < 0)

        # Compute success ratio (1 - failure rate)
        success_ratio = 1 - blitz_df["unsuccessful"].mean()

        return round(success_ratio, 3)





    

    def __repr__(self):
        return f"<Team {self.name}: {self.offensive_snaps()} offensive snaps>"


# Example usage
if __name__ == "__main__":
    kc = Team("KC", merged)

    #print(kc)
    """print("Offensive Snaps:", kc.offensive_snaps())
    print("\nOffensive Personnel:\n", kc.offensive_personnel_counts())
    print("\nFormations:\n", kc.formation_counts())
    print("\nDefensive Personnel Faced:\n", kc.defensive_personnel_faced())
    print("\nPressure Rate:", kc.pressure_rate())
    print(kc.success_rate_by_down())
    print(kc.success_rate_by_playType())
    print(kc.playType_by_down())"""
    #print("Blitz rate by down:\n", kc.blitz_rate_by_down())
    #print("KC success vs blitz:", kc.success_against_blitz())
    
