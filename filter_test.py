import nflreadpy as nfl
import pandas as pd

team_stats = nfl.load_team_stats(seasons=True)
team_stats_pandas = team_stats.to_pandas()

#print(team_stats_pandas.head())

#filename = input("Enter the filename (e.g., 'data/games_2024_0.csv'): ")

# pbp = pd.read_csv(filename)

# df = pd.DataFrame(pbp)

pos_team = input("Enter the offensive team abbreviation: ").upper()
def_team = input("Enter the defensive team abbreviation: ").upper()

filtered_teams = team_stats_pandas[team_stats_pandas['team'].isin([pos_team, def_team]) &
                                   team_stats_pandas['opponent_team'].isin([pos_team, def_team])]

#filtered_teams = team_stats_pandas[(team_stats_pandas['team'] == pos_team and team_stats_pandas['opponent_team'] == def_team) or 
                                   #(team_stats_pandas['team'] == def_team and team_stats_pandas['opponent_team'] == pos_team)]


print(filtered_teams)


# filtered_pos_team = df[df['posteam'] == pos_team]
# filtered_pos_team = df[df['defteam'] == def_team]

# filtered_pos_vs_def = df[(df['posteam'] == pos_team) & (df['defteam'] == def_team)]
pass_rating_columns = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions']

pass_rating_data = filtered_teams[pass_rating_columns]

print(pass_rating_data)



#print(filtered_pos_vs_def)
def calculate_offensive_pass_rating(group):
    cmp = group['completions']
    att = group['attempts']
    yds = group['passing_yards']
    td = group['passing_tds']
    interceptions = group['passing_interceptions']
    if att == 0: return 0.0
    a = max(0, min(((cmp / att) - 0.3) * 5, 2.375))
    b = max(0, min(((yds / att) - 3) * 0.25, 2.375))
    c = max(0, min((td / att) * 20, 2.375))
    d = max(0, min(2.375 - ((interceptions / att) * 25), 2.375))
    return ((a + b + c + d) / 6) * 100

# pass_rating_calc_data = pass_rating_data.apply(calculate_offensive_pass_rating)
# print(pass_rating_calc_data)
