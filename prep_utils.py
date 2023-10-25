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


def prepare_players_for_ranking(df_players, df_awards):
    df_playersnew = df_players.groupby(['playerID', 'year'])["stint"].idxmax()

    df_players = df_players.loc[df_playersnew]

    df_players["player_awards"] = 0

    df_players = player_award_count(df_players,df_awards, False)
    


    return df_players


def teams_agg(group):
    attributes = ['rank', 'playoff', 'o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 'o_oreb',
                      'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm',
                      'd_fga', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_oreb', 'd_dreb', 'd_reb', 'd_asts',
                      'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts', 'won', 'lost', 'W', 'L']
    
    min_stint_idx = group['stint'].astype(int).idxmin()
    result = {'tmID': group.loc[min_stint_idx, 'tmID']}
    
    for attr in attributes:
        result[attr] = group[attr].mean()
        
    return pd.Series(result)


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

def prepare_teams(teams_df, teams_post, past_years):
    print("Dropping divID in \033[1mTeams\033[0m...")

    teams_df.drop('divID', axis=1, inplace=True)
    
    print("Dropping ldID in \033[1mTeams\033[0m...")
    teams_df.drop('lgID', axis=1, inplace=True)

    print("Dropping seeded in \033[1mTeams\033[0m...")

    teams_df.drop('seeded', axis=1, inplace=True)

    print("Dropping tmORB, tmDRB, tmTRB, opptmORB, opptmDRB, opptmTRB in \033[1mTeams\033[0m...")

    teams_df.drop('tmORB', axis=1, inplace=True)
    teams_df.drop('tmDRB', axis=1, inplace=True)
    teams_df.drop('tmTRB', axis=1, inplace=True)
    teams_df.drop('opptmORB', axis=1, inplace=True)
    teams_df.drop('opptmDRB', axis=1, inplace=True)
    teams_df.drop('opptmTRB', axis=1, inplace=True)
    
    print("Dropping GP, homeW, homeL, awayW, awayL, confW, confL, min, attend, name, confID, franchID & arena in \033[1mTeams\033[0m...")

    teams_df.drop('GP', axis=1, inplace=True)
    teams_df.drop('homeW', axis=1, inplace=True)
    teams_df.drop('homeL', axis=1, inplace=True)
    teams_df.drop('awayW', axis=1, inplace=True)
    teams_df.drop('awayL', axis=1, inplace=True)
    teams_df.drop('confW', axis=1, inplace=True)
    teams_df.drop('confL', axis=1, inplace=True)
    teams_df.drop('min', axis=1, inplace=True)
    teams_df.drop('attend', axis=1, inplace=True)
    teams_df.drop('arena', axis=1, inplace=True)
    teams_df.drop('name', axis=1, inplace=True)
    teams_df.drop('franchID', axis=1, inplace=True)
    teams_df.drop('confID', axis=1, inplace=True)
    
    print("Creating attribute winrate \033[1mTeams\033[0m...")
    #teams_df["Winrate"] = teams_df["won"] / (teams_df["won"] + teams_df["lost"])
    
    print("Dropping won & lost in \033[1mTeams\033[0m...")
    #teams_df.drop('won', axis=1, inplace=True)
    #teams_df.drop('lost', axis=1, inplace=True)
    
    teams_df.drop('firstRound', axis=1, inplace=True)
    teams_df.drop('semis', axis=1, inplace=True)
    teams_df.drop('finals', axis=1, inplace=True)
    
    teams_post.drop('lgID',axis = 1, inplace = True)
    

    merged_df = pd.merge(teams_df, teams_post, on=['tmID', 'year'], how='left')
    merged_df['W'].fillna(0, inplace=True)
    merged_df['L'].fillna(0, inplace=True)
    
    merged_df = merged_df.sort_values(by=['tmID', 'year'])
    
    print("Converting Target PLAYOFF to binary on\033[1mTeams\033[0m...")
    merged_df['playoff'] = merged_df['playoff'].replace({'Y': 1, 'N': 0})
    
    def calculate_cumulative_sum(group):
        return group.shift(1).rolling(min_periods=1, window=past_years).sum().fillna(0)
    
    def calculate_cumulative_mean(group):
        return group.shift(1).rolling(min_periods=1, window=past_years).mean().fillna(0)
    
    merged_df = merged_df.sort_values(by=['tmID', 'year'])
    
    mean_attrs = ['rank', 'o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 'o_oreb',
                      'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm',
                      'd_fga', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_oreb', 'd_dreb', 'd_reb', 'd_asts',
                      'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts']
    
    sum_attrs =  ['won', 'lost', 'W', 'L']
    
    for attr in mean_attrs:
        merged_df[attr] = merged_df.groupby('tmID')[attr].transform(calculate_cumulative_mean)
    
    for attr in sum_attrs:
        merged_df[attr] = merged_df.groupby('tmID')[attr].transform(calculate_cumulative_sum)
    
    print("Creating attribute winrate \033[1mTeams\033[0m...")
    merged_df["Winrate"] = np.where((merged_df['won'] + merged_df['lost']) > 0,
                                       merged_df['won'] / (merged_df['won'] + merged_df['lost']),
                                       0)
    
    print("Creating attribute PlayOffs winrate \033[1mTeams\033[0m...")
    merged_df["PO_Winrate"] = np.where((merged_df['W'] + merged_df['L']) > 0,
                                       merged_df['W'] / (merged_df['W'] + merged_df['L']),
                                       0)
    
    merged_df.drop('W',axis = 1, inplace = True)
    merged_df.drop('L',axis = 1, inplace = True)
    merged_df.drop('won',axis = 1, inplace = True)
    merged_df.drop('lost',axis = 1, inplace = True)

    return merged_df
    


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

def feature_importance_players(df_players_info, df_players, df_teams):
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


    all_features = ['PPM', 'oRebounds', 'dRebounds', 'rebounds',
                'assists', 'steals', 'blocks', 'turnovers', 'PF', 'fg%', 'ft%',
                 '3pt%', 'dq', 'player_awards','PER']
 
    # %fg = fgMade/fgAttempted, %ft = ftMade/ftAttempted, %3pt = threeMade/threeAttempted, care for 0/0
    merged_df['PPM'] = np.where(merged_df['minutes'] == 0, 0, merged_df['points'] / merged_df['minutes'])
    merged_df['fg%'] = np.where(merged_df['fgAttempted'] == 0, 0, merged_df['fgMade'] / merged_df['fgAttempted'])
    merged_df['ft%'] = np.where(merged_df['ftAttempted'] == 0, 0, merged_df['ftMade'] / merged_df['ftAttempted'])
    merged_df['3pt%'] = np.where(merged_df['threeAttempted'] == 0, 0, merged_df['threeMade'] / merged_df['threeAttempted'])
    merged_df['playoff'] = np.where(merged_df['playoff'] == 'Y', 1, 0)    
    merged_df['PostPPM'] = np.where(merged_df['PostMinutes'] == 0, 0, merged_df['PostPoints'] / merged_df['PostMinutes'])
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

    return feature_importance, merged_df

def calculate_player_rating(row, feature_importance):
    position = row['pos']
    year = row['year']

    # Get feature importance for the player's position
    position_importance = feature_importance.get(position, {})

    # Sort features by importance and select the top 5
    top_features = sorted(position_importance, key=position_importance.get, reverse=True)[:5]
    negative = ['turnovers', 'dq', 'PF']
    top_features = [feature for feature in top_features if feature not in negative]
    negative_features = [feature for feature in top_features if feature in negative]


    # Calculate the player rating as the sum of the top 5 features
    rating = sum(row[feature] * position_importance[feature] for feature in top_features) - sum(row[feature] * position_importance[feature] for feature in negative_features)
 
    return rating

def ranking_players(feature_importance, df_new_players):
    df_new_players['rating'] = df_new_players.apply(calculate_player_rating, axis=1, args=(feature_importance,))
    df_new_players = df_new_players.sort_values(by='rating', ascending=False)
    
    df_new_players = df_new_players[['playerID', 'year', 'rating']]

    return df_new_players

def ranking_playoff_players(feature_importance, df_new_players):
    df_new_players['PostPER'] = ((df_new_players['PostfgMade'] * 85.910) + (df_new_players['PostSteals'] * 53.897) + (df_new_players['PostthreeMade'] * 51.757) + (df_new_players['PostftMade'] * 46.845) + (df_new_players['PostBlocks'] * 39.190) + (df_new_players['PostoRebounds'] * 39.190) + (df_new_players['PostAssists'] * 34.677) + (df_new_players['PostdRebounds'] * 14.707) - (df_new_players['PostPF'] * 17.174) - ((df_new_players['PostftAttempted'] - df_new_players['PostftMade']) * 20.091) - ((df_new_players['PostfgAttempted'] - df_new_players['PostfgMade']) * 39.190) - (df_new_players['PostTurnovers'] * 53.897)) * (np.where(df_new_players['PostMinutes'] == 0, 0, 1 / df_new_players['PostMinutes']))
    dic = {'GP': 'PostGP', 'GS': 'PostGS', 'minutes': 'PostMinutes', 'points': 'PostPoints', 'oRebounds': 'PostoRebounds', 'dRebounds': 'PostdRebounds', 'rebounds': 'PostRebounds', 'assists': 'PostAssists', 'steals': 'PostSteals', 'blocks': 'PostBlocks', 'turnovers': 'PostTurnovers', 'PF': 'PostPF', 'fg%': 'Postfg%', 'ft%': 'Postft%', '3pt%': 'Post3pt%', 'dq': 'PostDQ', 'player_awards': 'player_awards', 'PER': 'PostPER', 'PPM': 'PostPPM'}
    feature_importance_post = {}
    for position, features in feature_importance.items():
        feature_importance_post[position] = {}
        for feature, importance in features.items():
            feature_importance_post[position][dic[feature]] = importance
    

    # sort for player and then for year
    df_new_players['PostRating'] = df_new_players.apply(calculate_player_rating, axis=1, args=(feature_importance_post,))
    df_new_players = df_new_players.sort_values(by='playerID', ascending=False)

    df_new_players = df_new_players[['playerID', 'year', 'PostRating']]
    return df_new_players

def calculate_power_rating(group):
    # formula = (0.5 * player_rating + 0.5 * team_power_rating) / minutes

    # Calculate the sum of the player ratings using the formula above
    player_rating_sum = (0.5 * group['rating'] + 0.5 * group['PostRating']) / group['minutes']

    
    return player_rating_sum.sum()

def team_power_rating(df_teams, df_players):
 
    columns = ['playerID','year','rating','PostRating','pos','tmID','minutes']
    df_players = df_players[columns]
    merged_data = pd.merge(df_players, df_teams, on=['year', 'tmID'], how='inner')

    # Calculate the player's contribution to the team based on their Rating, PostRating, and minutes
    player_contributions = (0.5 * merged_data['rating'] + 0.5 * merged_data['PostRating']) / merged_data['minutes']
    
    # Group by year, team, and position, and calculate power rating
    team_power_ratings = player_contributions.groupby([merged_data['year'], merged_data['tmID'], merged_data['pos']]).sum().reset_index()
    team_power_ratings.columns = ['year', 'tmID', 'pos', 'PowerRating']
 
    # Merge the power ratings with the team DataFrame
    team_data = pd.merge(df_teams, team_power_ratings, on=['year', 'tmID'], how='left')

    # Pivot the table to have one row per team and columns for each position's power rating
    power_rating_pivot = team_data.pivot(index=['year', 'tmID'], columns='pos', values='PowerRating').reset_index()

    # Fill NaN values with 0 if necessary
    power_rating_pivot.fillna(0, inplace=True)

    # Calculate the overall power rating for each team by summing the position ratings
    position_columns = ['G', 'C-F', 'C', 'F', 'F-C', 'F-G', 'G-F']
    power_rating_pivot['OverallPowerRating'] = power_rating_pivot[position_columns].sum(axis=1)
    position_counts = (power_rating_pivot[position_columns] != 0).sum(axis=1)
    power_rating_pivot['OverallPowerRating'] /= position_counts.replace(0, 1)

    power_rating_pivot = pd.merge(power_rating_pivot, df_teams[['year', 'tmID', 'playoff']], on=['year', 'tmID'], how='left')
    
    return power_rating_pivot








