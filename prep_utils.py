import pandas as pd
import numpy as np


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
    

def prepare_coaches(df, df_awards):
    print("Dropping Attribute lgID in \033[1mCoaches\033[0m...")
    df.drop('lgID', axis=1, inplace=True)
    
    
    
    # Creating attribute coach_po_win_ratio, meaning the playoffs win ratio of a coach until the current year
    df = df.sort_values(by=['coachID', 'year'])
    df['total_playoffs_win'] = df.groupby('coachID')['post_wins'].cumsum() - df['post_wins']
    df['total_playoffs_lost'] = df.groupby('coachID')['post_losses'].cumsum() - df['post_losses']
    
    df['coach_po_win_ratio'] = np.where((df['total_playoffs_win'] + df['total_playoffs_lost']) > 0,
                                       df['total_playoffs_win'] / (df['total_playoffs_win'] + df['total_playoffs_lost']),
                                       0)

    df.drop('total_playoffs_win', axis=1, inplace=True)
    df.drop('total_playoffs_lost', axis=1, inplace=True)
    
    
    # Creating attribute playoffs_count, meaning the number of times a coach has gone to playoffs until the current year
    playoffs_mask = (df['post_wins'] != 0) | (df['post_losses'] != 0)
    df['playoffs_count'] = playoffs_mask.groupby(df['coachID']).cumsum() - playoffs_mask.astype(int)
    
    df.drop('post_wins', axis=1, inplace=True)
    df.drop('post_losses', axis=1, inplace=True)

    df["coach_awards"] = 0
    
    print("Creating attribute coach playoffs win ratio...")
    print("Creating attribute coach playoffs count...")
    
    df = coach_award_count(df,df_awards)
    
    print("Creating attribute num coach awards...")
    
    print("\n\033[1mCoaches Null Verification:\033[0m")
    print(df.isna().sum())
    
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
    teams_df.drop('tmTRB', axis=1, inplace=True)
    teams_df.drop('opptmORB', axis=1, inplace=True)
    teams_df.drop('opptmDRB', axis=1, inplace=True)
    teams_df.drop('opptmTRB', axis=1, inplace=True)

