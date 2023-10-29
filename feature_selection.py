import pandas as pd



def aggregate_stats_features(df):
    df_copy = df.copy()

    df_copy['o_fgw'] = df_copy['o_fgm'] / (df_copy['o_fga'])
    df_copy['o_ftw'] = df_copy['o_ftm'] / (df_copy['o_fta'])
    df_copy['o_3pw'] = df_copy['o_3pm'] / (df_copy['o_3pa'])
    
    df_copy = df_copy.drop("o_reb", axis=1)
    
    df_copy['d_fgw'] = df_copy['d_fgm'] / (df_copy['d_fga'])
    df_copy['d_ftw'] = df_copy['d_ftm'] / (df_copy['d_fta'])
    df_copy['d_3pw'] = df_copy['d_3pm'] / (df_copy['d_3pa'])
    
    df_copy = df_copy.drop("d_reb", axis=1)
    
    df_copy['p_fgWon'] = df_copy['p_fgMade'] / (df_copy['p_fgAttempted'])
    df_copy['p_ftWon'] = df_copy['p_ftMade'] / (df_copy['p_ftAttempted'])
    df_copy['p_threeWon'] = df_copy['p_threeMade'] / (df_copy['p_threeAttempted'])
    df_copy['p_PostfgWon'] = df_copy['p_PostfgMade'] / (df_copy['p_PostfgAttempted'])
    df_copy['p_PostftWon'] = df_copy['p_PostftMade'] / (df_copy['p_PostftAttempted'])
    df_copy['p_PostthreeWon'] = df_copy['p_PostthreeMade'] / (df_copy['p_PostthreeAttempted'])
    
    df_copy = df_copy.drop("p_PostRebounds", axis=1)
    df_copy = df_copy.drop("p_rebounds", axis=1)
    df_copy = df_copy.drop("p_GS", axis=1)
    df_copy = df_copy.drop("p_PostGS", axis=1)
    df_copy = df_copy.drop("p_minutes", axis=1)
    
    return df_copy