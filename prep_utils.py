import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import seaborn as sn
import matplotlib.pyplot as plt

def db_to_pandas(conn):
    df_awards = pd.read_sql_query("Select * from awards_players;",conn)
    df_coaches = pd.read_sql_query("Select * from coaches;",conn)
    df_players_teams = pd.read_sql_query("Select * from players_teams;",conn)
    df_players = pd.read_sql_query("Select * from players;",conn)
    df_series_post = pd.read_sql_query("Select * from series_post;",conn)
    df_teams_post = pd.read_sql_query("Select * from teams_post;",conn)
    df_teams = pd.read_sql_query("Select * from teams;",conn)
    
    df_awards.replace('', np.nan,inplace=True)
    df_coaches.replace('', np.nan,inplace=True)
    df_players_teams.replace('', np.nan,inplace=True)
    df_players.replace('', np.nan,inplace=True)
    df_series_post.replace('', np.nan,inplace=True)
    df_teams_post.replace('', np.nan,inplace=True)
    df_teams.replace('', np.nan,inplace=True)
    
    return [df_awards, df_coaches, df_players_teams, df_players, df_series_post, df_teams_post, df_teams]
    
def prepare_coaches(df, df_awards, past_years):
    print("Dropping Attribute lgID in \033[1mCoaches\033[0m...")
    df.drop('lgID', axis=1, inplace=True)
    
    def calculate_cumulative_sum(group):
        return group.shift(1).rolling(min_periods=1, window=past_years).sum().fillna(0)
    
    # Creating attribute coach_po_win_ratio, meaning the playoffs win ratio of a coach until the current year
    df = df.sort_values(by=['coachID', 'year'])
    df['reg_season_win'] = df.groupby('coachID')['won'].transform(calculate_cumulative_sum)
    df['reg_season_lost'] = df.groupby('coachID')['lost'].transform(calculate_cumulative_sum)
    
    
    df['total_playoffs_win'] = df.groupby('coachID')['post_wins'].transform(calculate_cumulative_sum)
    df['total_playoffs_lost'] = df.groupby('coachID')['post_losses'].transform(calculate_cumulative_sum)
    
    df['coach_po_win_ratio'] = np.where((df['total_playoffs_win'] + df['total_playoffs_lost']) > 0,
                                       df['total_playoffs_win'] / (df['total_playoffs_win'] + df['total_playoffs_lost']),
                                       0)
    
    df['coach_reg_win_ratio'] = np.where((df['reg_season_win'] + df['reg_season_lost']) > 0,
                                       df['reg_season_win'] / (df['reg_season_win'] + df['reg_season_lost']),
                                       0)

    df.drop('total_playoffs_win', axis=1, inplace=True)
    df.drop('total_playoffs_lost', axis=1, inplace=True)
    df.drop('reg_season_win', axis=1, inplace=True)
    df.drop('reg_season_lost', axis=1, inplace=True)
    
    
    playoffs_mask = (df['post_wins'] != 0) | (df['post_losses'] != 0)
    df['playoffs_count'] = playoffs_mask.groupby(df['coachID']).cumsum() - playoffs_mask.astype(int)


    df["coach_awards"] = 0
    
    print("Creating attribute coach previous regular season win ratio...")
    print("Creating attribute coach playoffs win ratio...")
    print("Creating attribute coach playoffs count...")
    print("Creating attribute coach awards count...")
    
    df = coach_award_count(df,df_awards)
    
    print("Dropping attribute post_wins..")
    print("Dropping attribute post_losses..")    
    print("Dropping attribute won..")    
    print("Dropping attribute lost..")    

    return df

def temp_prepare_coaches(df, df_awards):
    print("Dropping Attribute lgID in \033[1mCoaches\033[0m...")
    df.drop('lgID', axis=1, inplace=True)
    
    # Creating attribute coach_po_win_ratio, meaning the playoffs win ratio of a coach until the current year
    df = df.sort_values(by=['coachID', 'year'])
    df['reg_season_win'] = df.groupby('coachID')['won'].cumsum() - df['won']
    df['reg_season_lost'] = df.groupby('coachID')['lost'].cumsum() - df['lost']
    df['total_playoffs_win'] = df.groupby('coachID')['post_wins'].cumsum() - df['post_wins']
    df['total_playoffs_lost'] = df.groupby('coachID')['post_losses'].cumsum() - df['post_losses']
    
    df['coach_po_win_ratio'] = np.where((df['total_playoffs_win'] + df['total_playoffs_lost']) > 0,
                                       df['total_playoffs_win'] / (df['total_playoffs_win'] + df['total_playoffs_lost']),
                                       0)
    
    df['coach_reg_win_ratio'] = np.where((df['reg_season_win'] + df['reg_season_lost']) > 0,
                                       df['reg_season_win'] / (df['reg_season_win'] + df['reg_season_lost']),
                                       0)

    df.drop('total_playoffs_win', axis=1, inplace=True)
    df.drop('total_playoffs_lost', axis=1, inplace=True)
    df.drop('reg_season_win', axis=1, inplace=True)
    df.drop('reg_season_lost', axis=1, inplace=True)
    
    
    # Creating attribute playoffs_count, meaning the number of times a coach has gone to playoffs until the current year
    playoffs_mask = (df['post_wins'] != 0) | (df['post_losses'] != 0)
    df['playoffs_count'] = playoffs_mask.groupby(df['coachID']).cumsum() - playoffs_mask.astype(int)
    

    df["coach_awards"] = 0
    
    print("Creating attribute coach previous regular season win ratio...")
    print("Creating attribute coach playoffs win ratio...")
    print("Creating attribute coach playoffs count...")
    print("Creating attribute coach awards count...")
    
    df = coach_award_count(df,df_awards)
    
    print("Creating attribute num coach awards...")
    
    print("Dropping attribute post_wins..")
    print("Dropping attribute post_losses..")    
    print("Dropping attribute won..")    
    print("Dropping attribute lost..")    
    
    df.drop('post_wins', axis=1, inplace=True)
    df.drop('post_losses', axis=1, inplace=True)
    df.drop('won', axis=1, inplace=True)
    df.drop('lost', axis=1, inplace=True)
    
    print("\n\033[1mCoaches Null Verification:\033[0m")
    print(df.isna().sum())
    
    return df





def prepare_players_for_ranking(df_players, df_awards):

    
    df_playersnew = df_players.groupby(['playerID', 'year'])["stint"].idxmax()

    df_players = df_players.loc[df_playersnew]

    df_players["player_awards"] = 0

    df_players = player_award_count(df_players,df_awards, False)
    


    return df_players



def players_team_agg(group):
    attributes = ["GP","GS","minutes","points","oRebounds","dRebounds","rebounds","assists","steals","blocks","turnovers","PF","fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted","threeMade","dq","PostGP","PostGS","PostMinutes","PostPoints","PostoRebounds","PostdRebounds","PostRebounds","PostAssists","PostSteals","PostBlocks","PostTurnovers","PostPF","PostfgAttempted","PostfgMade","PostftAttempted","PostftMade","PostthreeAttempted","PostthreeMade","PostDQ"]
    
    min_stint_idx = group['stint'].astype(int).idxmin()
    result = {'tmID': group.loc[min_stint_idx, 'tmID']}
    
    for attr in attributes:
        result[attr] = group[attr].sum()
        
    return pd.Series(result)


def prepare_player_teams(df,df_awards,past_years):
    print("Dropping Attribute lgID in \033[1mCoaches\033[0m...")
    df.drop('lgID', axis=1, inplace=True)
    
    df = df.groupby(['playerID', 'year']).apply(players_team_agg).reset_index()
    
    def calculate_cumulative_sum(group):
        return group.shift(1).rolling(min_periods=1, window=past_years).mean().fillna(0)
    
    df = df.sort_values(by=['playerID', 'year'])
    
    attributes = ["GP","GS","minutes","points","oRebounds","dRebounds","rebounds","assists","steals","blocks","turnovers","PF","fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted","threeMade","dq","PostGP","PostGS","PostMinutes","PostPoints","PostoRebounds","PostdRebounds","PostRebounds","PostAssists","PostSteals","PostBlocks","PostTurnovers","PostPF","PostfgAttempted","PostfgMade","PostftAttempted","PostftMade","PostthreeAttempted","PostthreeMade","PostDQ"]
    
    for attr in attributes:
        
        df[attr] = df.groupby('playerID')[attr].transform(calculate_cumulative_sum)
    
    df["player_awards"] = 0
    
    df = player_award_count(df,df_awards, True)
    
    
    print(df.to_string())
    
    return df


def player_award_count(df,df_awards,previous_years):
    for index, row in df_awards.iterrows():
        player_id = row['playerID']
        award = row['award']
        award_year = row['year']
        if(previous_years):
            mask = (df['playerID'] == player_id) & (df['year'] > award_year)
        else:
            mask = (df['playerID'] == player_id) & (df['year'] == award_year)
        df.loc[mask, 'player_awards'] += 1
    
    return df
      
    
def coach_award_count(df,df_awards):
    for index, row in df_awards.iterrows():
        coach_id = row['playerID']
        award = row['award']
        award_year = row['year']
        
        if 'Coach' in award:
            mask = (df['coachID'] == coach_id) & (df['year'] > award_year)
            df.loc[mask, 'coach_awards'] += 1
    
    return df

def prepare_players(df, df_players_teams):
    print("\nRemoving players from \033[1mPlayers\033[0m without any single played game...")
    df = df[df['bioID'].isin(df_players_teams['playerID'])]
    
    print("Dropping Attributes firstseason & lastseason in \033[1mPlayers\033[0m...")
    df = df.drop('firstseason', axis=1)
    df = df.drop('lastseason', axis=1)
    
    print("\n\033[1mPlayers Null Verification:\033[0m")
    print(df.isna().sum())
    
    df = df.drop('collegeOther', axis=1)
    
    num_weight = (df['weight'] == 0).sum()
    num_height = (df['height'] == 0).sum()
    
    print(f"\n\033[1mNull Entries weight\033[0m - {num_weight}")
    print(f"\033[1mNull Entries height\033[0m - {num_height}")
    
    average_weight = df[df['weight'] != 0]['weight'].mean()
    average_height = df[df['height'] != 0]['height'].mean()
    
    df['weight'] = df['weight'].replace(0, average_weight)
    df['height'] = df['height'].replace(0, average_height)


    num_weight = (df['weight'] == 0).sum()
    num_height = (df['height'] == 0).sum()
    
    print("\n\033[1mReplacing Height & Weight Null Values by its average...\033[0m")
    print(f"\033[1mNull Entries weight\033[0m - {num_weight}")
    print(f"\033[1mNull Entries height\033[0m - {num_height}")

    return df


def best_players(player_teams_df, teams_df):
    # Filter the teams DataFrame to contain only playoff appearances
    playoff_teams = teams_df[(teams_df['playoff'] == 'Y')]

   
    # Merge the players and playoff_teams DataFrames on 'lgID' and 'year'
    merged_df = pd.merge(player_teams_df, playoff_teams, left_on=['tmID', 'year'], right_on=['tmID', 'year'], how='inner')
 
    # Table with the count of appearances for each player
    playoff_count = merged_df['playerID'].value_counts()
 
    return playoff_count

def best_colleges(player_teams_df, teams_df, players_df):
    
    playoff_apperances = best_players(player_teams_df, teams_df)
   
    
    # Merge the merged_df with the players DataFrame to get college information
    merged_df = pd.merge(playoff_apperances, players_df, left_on='playerID', right_on='bioID', how='inner')

 
    # Group by college and sum playoff appearances
    college_playoff_sum = merged_df.groupby('college')['bioID'].count().reset_index()

    # Sort the values of playoff_count
    college_playoff_sum = college_playoff_sum.sort_values(by='bioID', ascending=False)

    college_playoff_sum.columns = ['college', 'TotalPlayoffAppearances']

    # Add a new column for the college ranking
    college_playoff_sum['CollegeRank'] = college_playoff_sum['TotalPlayoffAppearances'].rank(ascending=False, method='dense').astype(int)

    return college_playoff_sum

def prepare_teams(teams_df):
    print("Dropping divID in \033[1mTeams\033[0m...")

    teams_df.drop('divID', axis=1, inplace=True)
    
    print("Dropping ldID in \033[1mTeams\033[0m...")
    teams_df.drop('lgID', axis=1, inplace=True)

    print("Dropping seeded in \033[1mTeams\033[0m...")

    teams_df.drop('seeded', axis=1, inplace=True)

    print("Dropping tmORB, tmDRB, tmTRB, opptmORB, opptmDRB, opptmTRB in \033[1mTeams\033[0m...")

    teams_df.drop('tmORB', axis=1, inplace=True)
    teams_df.drop('tmDRB', axis=1, inplace=True)
    teams_df.drop('seeded', axis=1, inplace=True)
    teams_df.drop('tmTRB', axis=1, inplace=True)
    teams_df.drop('opptmORB', axis=1, inplace=True)
    teams_df.drop('opptmDRB', axis=1, inplace=True)
    teams_df.drop('opptmTRB', axis=1, inplace=True)


def custom_agg(group):
    min_stint_coach = group.loc[group['stint'].idxmin()]
    #coach_info = {Nome,WinRate Normal,WinRate Playoffs, Nr Playoffs, Nr Awards}
    coach_info = f"{min_stint_coach['coachID']},{min_stint_coach['coach_reg_win_ratio']},{min_stint_coach['coach_po_win_ratio']},{min_stint_coach['playoffs_count']},{min_stint_coach['coach_awards']}"

    return pd.Series({
        'coachID': min_stint_coach['coachID'],
        'stint': min_stint_coach['stint'],
        'coach_reg_season_wr': min_stint_coach['coach_reg_win_ratio'],
        'coach_po_season_wr': min_stint_coach['coach_po_win_ratio'],
        'coach_playoffs_count': min_stint_coach['playoffs_count'],
        'coach_awards': min_stint_coach['coach_awards'],   
    })
    

def group_coaches(df):
    new_df = df.groupby(['year', 'tmID']).apply(custom_agg).reset_index()
    new_df.drop('stint', axis=1, inplace=True)
    
    print("\n\033[1mCoaches Null Verification:\033[0m")
    print(new_df.isna().sum())
    
    return new_df

def ranking_players(df_players_info, df_players, df_teams):
    # Merge the DataFrames
    merged_df = pd.merge(df_players_info, df_players, left_on='playerID', right_on='bioID', how='inner')


    merged_df = pd.merge(merged_df, df_teams[['tmID', 'year', 'playoff']], left_on=['tmID', 'year'], right_on=['tmID', 'year'], how='inner')
 

    position_features = {
    'G': ['points', 'assists', 'steals', 'turnovers', 'minutes'],
    'C-F': ['points', 'assists', 'steals', 'turnovers', 'minutes'],
    'C': ['points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes'],
    'F': ['points', 'rebounds', 'blocks', 'minutes'],
    'F-C': ['points', 'rebounds', 'blocks', 'minutes'],
    'F-G': ['points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes'],
    'G-F': ['points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes']
    }


    all_features = ['minutes', 'points', 'oRebounds', 'dRebounds', 'rebounds',
                'assists', 'steals', 'blocks', 'turnovers', 'PF', 'fg%', 'ft%',
                 '3pt%', 'dq', 'player_awards','PER']
 
    # %fg = fgMade/fgAttempted, %ft = ftMade/ftAttempted, %3pt = threeMade/threeAttempted, care for 0/0
    merged_df['fg%'] = np.where(merged_df['fgAttempted'] == 0, 0, merged_df['fgMade'] / merged_df['fgAttempted'])
    merged_df['ft%'] = np.where(merged_df['ftAttempted'] == 0, 0, merged_df['ftMade'] / merged_df['ftAttempted'])
    merged_df['3pt%'] = np.where(merged_df['threeAttempted'] == 0, 0, merged_df['threeMade'] / merged_df['threeAttempted'])
    merged_df['playoff'] = np.where(merged_df['playoff'] == 'Y', 1, 0)    
    merged_df['Postfg%'] = np.where(merged_df['PostfgAttempted'] == 0, 0, merged_df['PostfgMade'] / merged_df['PostfgAttempted'])
    merged_df['Postft%'] = np.where(merged_df['PostftAttempted'] == 0, 0, merged_df['PostftMade'] / merged_df['PostftAttempted'])
    merged_df['Post3pt%'] = np.where(merged_df['PostthreeAttempted'] == 0, 0, merged_df['PostthreeMade'] / merged_df['PostthreeAttempted'])

    merged_df['PER'] = ((merged_df['fgMade'] * 85.910) + (merged_df['steals'] * 53.897) + (merged_df['threeMade'] * 51.757) + (merged_df['ftMade'] * 46.845) + (merged_df['blocks'] * 39.190) + (merged_df['oRebounds'] * 39.190) + (merged_df['assists'] * 34.677) + (merged_df['dRebounds'] * 14.707) - (merged_df['PF'] * 17.174) - ((merged_df['ftAttempted'] - merged_df['ftMade']) * 20.091) - ((merged_df['fgAttempted'] - merged_df['fgMade']) * 39.190) - (merged_df['turnovers'] * 53.897)) * (np.where(merged_df['minutes'] == 0, 0, 1 / merged_df['minutes']))


    # Create an empty dictionary to store feature importance for each position
    feature_importance = {position: {} for position in position_features.keys()}
    df_players = merged_df.copy()

    #Change the playoff column to 1 and 0
   
    # Iterate over positions and build models
    for position, features in position_features.items():
        # Prepare the data for the current position
        position_data = df_players[df_players['pos'] == position]
        X = position_data[all_features]
        y = position_data['playoff']  # Define your ranking metric (e.g., player impact score)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Calculate feature importance
        feature_importance[position] = dict(zip(all_features, model.feature_importances_))

        # Evaluate the model (you may use different metrics depending on your ranking metric)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"\nMean Squared Error for {position}: {mse}")

        # Print feature importance for each position sorted
        print(f"Feature importance for {position}:")
        for feature, importance in sorted(feature_importance[position].items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance}")


