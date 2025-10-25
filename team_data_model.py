# Columns: game_id , season_type , home_team , away_team , posteam , defteam , location , 
# stadium , roof , surface , temp , wind , qtr , down , ydstogo , yardline_100 , time , 
# quarter_seconds_remaining , game_seconds_remaining , score_differential , goal_to_go , 
# posteam_timeouts_remaining , defteam_timeouts_remaining , play_type , rush_attempt , pass_attempt , 
# field_goal_attempt , shotgun , no_huddle , qb_dropback , run_location , run_gap , pass_length , 
# pass_location , air_yards , yards_after_catch , yards_gained , success , epa , touchdown
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist

current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct full path to data file
data_path = os.path.join(current_dir, 'data', 'pbp_2020_0_cleaned.csv')
df = pd.read_csv(data_path)

def Model(home_team , away_team , posteam , defteam , location , 
          stadium , roof , surface , temp , wind , qtr , down , ydstogo , yardline_100 , 
          time , quarter_seconds_remaining , game_seconds_remaining , score_differential , 
          goal_to_go ,  posteam_timeouts_remaining , defteam_timeouts_remaining , play_type , 
          rush_attempt , pass_attempt ,  field_goal_attempt , shotgun , no_huddle , 
          qb_dropback , run_location , run_gap , pass_length , pass_location , air_yards , 
          yards_after_catch , yards_gained , success , epa , touchdown):
    
    # I want to get a similarity score. In other words, using the pbp data from previous years, 
    # I want to find the plays that are the most similar, order tghm by epa then suggest the plays 
    # with the best epa that are the most similar
    #Ex: Similarirty Score: 50-60%, Best EPA: 11.5,10, 9.5
    #Similarirty Score: 60-70%, Best EPA: 10.3, 9.8, 9.5 etc
    home_team_enc = pd.get_dummies(df['home_team'], prefix='home_team')
    away_team_enc = pd.get_dummies(df['away_team'], prefix='away_team')
    posteam_enc = pd.get_dummies(df['posteam'], prefix='posteam')
    defteam_enc = pd.get_dummies(df['defteam'], prefix='defteam')

    #I want to use season_type , home_team , away_team , posteam , defteam , location , 
    # stadium , roof , surface , temp , wind , qtr , down , ydstogo , yardline_100 , 
    # time , quarter_seconds_remaining , game_seconds_remaining , score_differential , goal_to_go  to define similarity
    feature_cols = ['season_type', 'location', 'stadium', 'roof', 'surface', 'temp', 'wind', 'qtr', 'down', 
                    'ydstogo', 'yardline_100', 'time', 'quarter_seconds_remaining', 'game_seconds_remaining', 'score_differential', 'goal_to_go']
    #for each feauture in feature_cols, I want to get the corresponding column from the input parameters. if it's the same the score is 1 else 0
    
    


    # This function would contain the logic to predict the outcome of a play
    # based on the input features. For now, it returns a dummy value.
    return 0.0