import pandas as pd

df = pd.read_csv("data/pbp_2020_0.csv")

# picking whatever variables (columns) you want
variables = ["play_id", "game_id", "play_type", "yardline_100", "quarter_end", "drive"]   # change this needed
subset = df[variables]

print(subset)   # show first 5 rows

