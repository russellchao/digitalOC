# Extract essential data from specific plays 
# Testing using pbp_2024_0.csv file right now

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Open the 2024 Play-by-Play CSV file (First Part)
df = pd.read_csv("Data/pbp_2024_0.csv")
print(df.head())

'''
x variables include:
1. Game Situation Variables (Critical)
down - Current down (1st, 2nd, 3rd, 4th)
ydstogo - Yards to go for first down
yardline_100 - Distance to opponent's end zone (field position)
goal_to_go - Binary flag if in goal-to-go situation
quarter_seconds_remaining - Time left in current quarter
game_seconds_remaining - Total time left in game
game_half - Half1 or Half2
2. Score and Win Probability Variables (Critical)
score_differential - Current point differential (positive = leading)
wp - Win probability (0-1)
vegas_wp - Vegas-adjusted win probability
ep - Expected points for current situation
posteam_timeouts_remaining - Timeouts left for offensive team
defteam_timeouts_remaining - Timeouts left for defensive team
3. Team and Matchup Variables (Important)
posteam - Possession team (offensive team)
defteam - Defensive team
posteam_type - Home or away team on offense
div_game - Divisional game flag
4. Formation and Personnel Variables (Important)
shotgun - Binary flag for shotgun formation
no_huddle - Binary flag for no-huddle offense
qb_dropback - Binary flag indicating QB dropped back
5. Environmental Variables (Moderately Important)
roof - Stadium type (outdoors, dome, retractable)
surface - Field surface (grass, turf)
temp - Temperature in Fahrenheit
wind - Wind speed in mph
weather - Weather description
6. Drive Context Variables (Moderately Important)
drive_play_count - Number of plays in current drive
drive_time_of_possession - Time of possession for current drive
drive_start_yard_line - Starting field position of drive
ydsnet - Net yards gained on current drive
7. Historical Performance Variables (Advanced)
total_home_rush_epa / total_away_rush_epa - Cumulative rushing EPA
total_home_pass_epa / total_away_pass_epa - Cumulative passing EPA
series_success - Success rate in current series
cp - Completion probability (for pass plays)
cpoe - Completion percentage over expected
8. Coaching and Personnel Variables (Useful)
home_coach / away_coach - Head coaches
passer_player_name - Starting QB
season_type - Regular season, playoffs, etc.
week - Week of the season
9. Betting Market Variables (Contextual)
spread_line - Point spread
total_line - Over/under total
vegas_home_wp - Vegas home team win probability
'''

'''
y variables include:
play_type, run_location, pass_length, pass_type
'''

# We need 80% of the data for training (X_train, y_train), 20% for testing (X_test, y_test)

y = df[['play_type', 'run_location', 'pass_length', 'pass_location']]
print(y.head(50))

# x variables using categories 1-3 for now
x = df[['down', 'ydstogo', 'yardline_100', 'goal_to_go', 'quarter_seconds_remaining',
        'game_seconds_remaining', 'game_half', 'score_differential', 'wp', 'vegas_wp',
        'ep', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'posteam', 'defteam',
        'posteam_type', 'div_game']]
print(x.head(50))


def train_model(case, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    print(x_train)
    print(x_test)

    ''' MODEL BUILDING '''

    # Linear Regression
    lr = LinearRegression()
    lr.fit(x_train, y_train)














def print_results(case):
    print(f"Down: {case[0][0]}")
    print(f"Yards to go: {case[0][1]}")
    print(f"Distance to end zone: {case[0][2]}")
    print(f"Goal to go: {case[0][3]}")
    print(f"Quarter seconds remaining: {case[0][4]}")  
    print(f"Game seconds remaining: {case[0][5]}")
    print(f"Game half: {case[0][6]}")
    print(f"Score differential: {case[0][7]}")
    print(f"Win probability: {case[0][8]}")
    print(f"Vegas win probability: {case[0][9]}")
    print(f"Expected points: {case[0][10]}")
    print(f"Offensive team timeouts remaining: {case[0][11]}")
    print(f"Defensive team timeouts remaining: {case[0][12]}")
    print(f"Offensive team: {case[0][13]}")
    print(f"Defensive team: {case[0][14]}")
    print(f"Offensive team type: {case[0][15]}")
    print(f"Divisional game: {case[0][16]}")


# FOR LATER: Test the model with random x variables
test_case_1 = [[2, 5, 30, 0, 300, 900, 1, -3, 0.45, 0.5, 1.2, 2, 2, 'NE', 'NYG', 'home', 0]]
test_case_2 = [[3, 8, 50, 1, 120, 600, 2, 7, 0.65, 0.6, 3.5, 1, 1, 'KC', 'DEN', 'away', 1]]
test_case_3 = [[1, 10, 80, 0, 900, 1800, 1, 0, 0.5, 0.55, 0.0, 3, 3, 'DET', 'BUF', 'home', 0]]
test_case_4 = [[4, 1, 5, 1, 30, 120, 4, -10, 0.2, 0.3, -2.5, 0, 0, 'PHI', 'DAL', 'away', 1]] # PHI would presumably run the "tush-push" play in this situation


# print_results(test_case_1)
# print_results(test_case_2)
# print_results(test_case_3)
# print_results(test_case_4)