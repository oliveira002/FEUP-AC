import numpy as np
import pandas as pd

def coach_award_count(df,df_awards):
    for index, row in df_awards.iterrows():
        coach_id = row['playerID']
        award = row['award']
        award_year = row['year']
        
        if 'Coach' in award:
            mask = (df['coachID'] == coach_id) & (df['year'] > award_year)
            df.loc[mask, 'coach_awards'] += 1
    
    return df


# Raw Prepare Coaches
def group_coaches(df):
    new_df = df.groupby(['year', 'tmID']).apply(raw_coach_agg).reset_index()
    new_df.drop('stint', axis=1, inplace=True)
    
    print("\n\033[1mCoaches Null Verification:\033[0m")
    print(new_df.isna().sum())
    
    return new_df


def raw_coach_agg(group):
    min_stint_coach = group.loc[group['stint'].idxmin()]
   
    return pd.Series({
        'coachID': min_stint_coach['coachID'],
        'stint': min_stint_coach['stint'],
        'total_reg_season_win': min_stint_coach['total_reg_season_win'],
        'total_reg_season_lost': min_stint_coach['total_reg_season_lost'],
        'total_playoffs_win': min_stint_coach['total_playoffs_win'],
        'total_playoffs_lost': min_stint_coach['total_playoffs_lost'],
        'coach_playoffs_count': min_stint_coach['playoffs_count'],
        'coach_awards': min_stint_coach['coach_awards'],   
    })

def raw_prepare_coaches(df, df_awards, past_years):
    df_copy = df.copy()
    print("Dropping Attribute lgID in \033[1mCoaches\033[0m...")
    df_copy.drop('lgID', axis=1, inplace=True)
    
    def calculate_cumulative_sum(group):
        return group.shift(1).rolling(min_periods=1, window=past_years).sum().fillna(0) 
    
    # Creating attribute coach_po_win_ratio, meaning the playoffs win ratio of a coach until the current year
    df_copy = df_copy.sort_values(by=['coachID', 'year'])
    df_copy['total_reg_season_win'] = df_copy.groupby('coachID')['won'].transform(calculate_cumulative_sum)
    df_copy['total_reg_season_lost'] = df_copy.groupby('coachID')['lost'].transform(calculate_cumulative_sum)
    
    
    df_copy['total_playoffs_win'] = df_copy.groupby('coachID')['post_wins'].transform(calculate_cumulative_sum)
    df_copy['total_playoffs_lost'] = df_copy.groupby('coachID')['post_losses'].transform(calculate_cumulative_sum)
    
    playoffs_mask = (df_copy['post_wins'] != 0) | (df_copy['post_losses'] != 0)
    df_copy['playoffs_count'] = playoffs_mask.groupby(df_copy['coachID']).cumsum() - playoffs_mask.astype(int)


    df_copy["coach_awards"] = 0
    
    print("Creating attribute coach previous regular season win ratio...")
    print("Creating attribute coach playoffs win ratio...")
    print("Creating attribute coach playoffs count...")
    print("Creating attribute coach awards count...")
    
    df_copy = coach_award_count(df_copy,df_awards)


    return df_copy

# Raw Prepare Players Teams
def group_players_stats_by_team(df_players):
    df_copy = df_players.copy()
    df_copy.drop('playerID', axis=1, inplace=True)
    result_df = df_copy.groupby(['tmID', 'year']).sum().reset_index()
    result_df.columns = ['tmID', 'year'] + [f'players_{col}' for col in result_df.columns[2:]]
    return result_df

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

def players_team_agg(group):
    attributes = ["GP","GS","minutes","points","oRebounds","dRebounds","rebounds","assists","steals","blocks","turnovers","PF","fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted","threeMade","dq","PostGP","PostGS","PostMinutes","PostPoints","PostoRebounds","PostdRebounds","PostRebounds","PostAssists","PostSteals","PostBlocks","PostTurnovers","PostPF","PostfgAttempted","PostfgMade","PostftAttempted","PostftMade","PostthreeAttempted","PostthreeMade","PostDQ"]
    
    min_stint_idx = group['stint'].astype(int).idxmin()
    result = {'tmID': group.loc[min_stint_idx, 'tmID']}
    
    for attr in attributes:
        result[attr] = group[attr].sum()
        
    return pd.Series(result)

def raw_prepare_player_teams(df,df_awards,past_years):
    df_copy = df.copy()
    print("Dropping Attribute lgID in \033[1mPlayers_Teams\033[0m...")
    df_copy.drop('lgID', axis=1, inplace=True)
    
    df_copy = df_copy.groupby(['playerID', 'year']).apply(players_team_agg).reset_index()
    
    def calculate_cumulative_mean(group):
        return group.shift(1).rolling(min_periods=1, window=past_years).mean().fillna(0)
    
    df_copy = df_copy.sort_values(by=['playerID', 'year'])
    
    attributes = ["GP","GS","minutes","points","oRebounds","dRebounds","rebounds","assists","steals","blocks","turnovers","PF","fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted","threeMade","dq","PostGP","PostGS","PostMinutes","PostPoints","PostoRebounds","PostdRebounds","PostRebounds","PostAssists","PostSteals","PostBlocks","PostTurnovers","PostPF","PostfgAttempted","PostfgMade","PostftAttempted","PostftMade","PostthreeAttempted","PostthreeMade","PostDQ"]
    
    for attr in attributes:
        
        df_copy[attr] = df_copy.groupby('playerID')[attr].transform(calculate_cumulative_mean)
    
    df_copy["player_awards"] = 0
    
    df_copy = player_award_count(df_copy,df_awards, True)
    
    return df_copy

# Raw Prepare Teams
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

def raw_prepare_teams(teams_df, teams_post, past_years):
    df_copy = teams_df.copy()
    df_post_copy = teams_post.copy()
    print("Dropping divID in \033[1mTeams\033[0m...")

    df_copy.drop('divID', axis=1, inplace=True)
    
    print("Dropping ldID in \033[1mTeams\033[0m...")
    df_copy.drop('lgID', axis=1, inplace=True)

    print("Dropping seeded in \033[1mTeams\033[0m...")

    df_copy.drop('seeded', axis=1, inplace=True)

    print("Dropping tmORB, tmDRB, tmTRB, opptmORB, opptmDRB, opptmTRB in \033[1mTeams\033[0m...")

    df_copy.drop('tmORB', axis=1, inplace=True)
    df_copy.drop('tmDRB', axis=1, inplace=True)
    df_copy.drop('tmTRB', axis=1, inplace=True)
    df_copy.drop('opptmORB', axis=1, inplace=True)
    df_copy.drop('opptmDRB', axis=1, inplace=True)
    df_copy.drop('opptmTRB', axis=1, inplace=True)
    
    df_post_copy.drop('lgID',axis = 1, inplace = True)
    
    merged_df = pd.merge(df_copy, df_post_copy, on=['tmID', 'year'], how='left')
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
    
    mean_attrs = ['min', 'GP', 'rank', 'o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 'o_oreb',
                      'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm',
                      'd_fga', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_oreb', 'd_dreb', 'd_reb', 'd_asts',
                      'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts']
    
    sum_attrs =  ['won', 'lost', 'W', 'L','homeW', 'homeL', 'awayW', 'awayL', 'confW', 'confL']
    
    for attr in mean_attrs:
        merged_df[attr] = merged_df.groupby('tmID')[attr].transform(calculate_cumulative_mean)
    
    for attr in sum_attrs:
        merged_df[attr] = merged_df.groupby('tmID')[attr].transform(calculate_cumulative_sum)
        

    playoffs_mask = (merged_df['playoff'] != 0)
    merged_df['team_playoffs_count'] = playoffs_mask.groupby(df_copy['tmID']).cumsum() - playoffs_mask.astype(int)
    
    print("Creating attribute winrate \033[1mTeams\033[0m...")
    merged_df["Winrate"] = np.where((merged_df['won'] + merged_df['lost']) > 0,
                                       merged_df['won'] / (merged_df['won'] + merged_df['lost']),
                                       0)
    
    print("Creating attribute PlayOffs winrate \033[1mTeams\033[0m...")
    merged_df["PO_Winrate"] = np.where((merged_df['W'] + merged_df['L']) > 0,
                                       merged_df['W'] / (merged_df['W'] + merged_df['L']),
                                       0)
    
    return merged_df

def group_players_stats_by_team(df_players):
    df_copy = df_players.copy()
    df_copy.drop('playerID', axis=1, inplace=True)
    result_df = df_copy.groupby(['tmID', 'year']).sum().reset_index()
    result_df.columns = ['tmID', 'year'] + [f'players_{col}' for col in result_df.columns[2:]]
    return result_df

def merge_all_raw_data(df_teams,df_players_teams,df_coaches,df_awards,df_post_teams,num_years):
    df_new_coaches = raw_prepare_coaches(df_coaches, df_awards,num_years)
    df_new_coaches = group_coaches(df_new_coaches)
    
    df_new_players_teams = raw_prepare_player_teams(df_players_teams,df_awards,num_years)
    
    df_new_players_teams = group_players_stats_by_team(df_new_players_teams)
    
    df_new_teams = raw_prepare_teams(df_teams,df_post_teams,num_years)

    merged_df = pd.merge(df_new_teams, df_new_coaches, on=['tmID', 'year'], how='left')
    merged_df = pd.merge(merged_df, df_new_players_teams, on=['tmID', 'year'], how='left')
    
    
    return merged_df
    