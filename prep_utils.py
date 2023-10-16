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
