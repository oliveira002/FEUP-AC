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

    return [df_awards, df_coaches, df_players_teams, df_players, df_series_post, df_teams_post, df_teams]
    

def prepare_coaches(df, df_awards):
    print("Dropping Attribute lgID in \033[1mCoaches\033[0m...")
    df.drop('lgID', axis=1, inplace=True)
    df = df.sort_values(by=['coachID', 'year'])

    df['total_playoffs_win'] = df.groupby('coachID')['post_wins'].cumsum() - df['post_wins']
    df['total_playoffs_lost'] = df.groupby('coachID')['post_losses'].cumsum() - df['post_losses']
    
    
    df['coach_po_win_ratio'] = np.where((df['total_playoffs_win'] + df['total_playoffs_lost']) > 0,
                                       df['total_playoffs_win'] / (df['total_playoffs_win'] + df['total_playoffs_lost']),
                                       0)

    df.drop('total_playoffs_win', axis=1, inplace=True)
    df.drop('total_playoffs_lost', axis=1, inplace=True)
    
    df["coach_awards"] = 0
    
    print("Creating attribute coach playoffs win ratio...")
    
    df = coach_award_count(df,df_awards)
    
    print("Creating attribute num coach awards...")
    
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
    print("Removing players from \033[1mPlayers\033[0m without any single played game...")
    df = df[df['bioID'].isin(df_players_teams['playerID'])]
    
    print("Dropping Attributes firstseason & lastseason in \033[1mPlayers\033[0m...")
    df.drop('lgID', axis=1, inplace=True)
    return df
