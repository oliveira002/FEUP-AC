import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from scipy.stats import pointbiserialr
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



def aggregate_stats_features(df):
    df_copy = df.copy()
    
    df_copy['o_fgw'] = np.where(df_copy['o_fga'] > 0, df_copy['o_fgm'] / df_copy['o_fga'], 0)
    df_copy['o_ftw'] = np.where(df_copy['o_fta'] > 0, df_copy['o_ftm'] / df_copy['o_fta'], 0)
    df_copy['o_3pw'] = np.where(df_copy['o_3pa'] > 0, df_copy['o_3pm'] / df_copy['o_3pa'], 0)
    
    df_copy = df_copy.drop("o_reb", axis=1)
    
    df_copy['d_fgw'] = np.where(df_copy['d_fga'] > 0, df_copy['d_fgm'] / df_copy['d_fga'], 0)
    df_copy['d_ftw'] = np.where(df_copy['d_fta'] > 0, df_copy['d_ftm'] / df_copy['d_fta'], 0)
    df_copy['d_3pw'] = np.where(df_copy['d_3pa'] > 0, df_copy['d_3pm'] / df_copy['d_3pa'], 0)
    
    df_copy = df_copy.drop("d_reb", axis=1)
    
    df_copy['p_fgWon'] = np.where(df_copy['p_fgAttempted'] > 0, df_copy['p_fgMade'] / df_copy['p_fgAttempted'], 0)
    df_copy['p_ftWon'] = np.where(df_copy['p_ftAttempted'] > 0, df_copy['p_ftMade'] / df_copy['p_ftAttempted'], 0)
    df_copy['p_threeWon'] = np.where(df_copy['p_threeAttempted'] > 0, df_copy['p_threeMade'] / df_copy['p_threeAttempted'], 0)
    df_copy['p_PostfgWon'] = np.where(df_copy['p_PostfgAttempted'] > 0, df_copy['p_PostfgMade'] / df_copy['p_PostfgAttempted'], 0)
    df_copy['p_PostftWon'] = np.where(df_copy['p_PostftAttempted'] > 0, df_copy['p_PostftMade'] / df_copy['p_PostftAttempted'], 0)
    df_copy['p_PostthreeWon'] = np.where(df_copy['p_PostthreeAttempted'] > 0, df_copy['p_PostthreeMade'] / df_copy['p_PostthreeAttempted'], 0)
    
    df_copy = df_copy.drop(["p_PostRebounds", "p_rebounds", "p_GS", "p_PostGS", "p_minutes"], axis=1)
    
    return df_copy

def model_with_pca(model,x_train,x_test,y_train,y_test,n_components):
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    
    model.fit(x_train_pca, y_train)
    y_pred_pca = model.predict(x_test_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
            

def model_with_rfe(model,x_train,x_test,y_train,y_test,n_features):
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    x_train_rfe = rfe.fit_transform(x_train, y_train)
    x_test_rfe = rfe.transform(x_test)
    
    model.fit(x_train_rfe, y_train)
    y_pred_rfe = model.predict(x_test_rfe)
    accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
    

def lasso_feature_selection(x_train,x_test,y_train,y_test,alpha):
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(x_train, y_train)
    y_pred = lasso_model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
  
    # Print the coefficients of the selected features
    selected_features = [feature for feature, coef in zip(x_train.columns, lasso_model.coef_) if coef != 0]
    print(f"Selected Features: {selected_features}")
        

def fs_teams(df_teams):
    df_copy = df_teams.copy()
    attributes = ['o_fg', 'o_ft', 'o_3p', 'd_fg', 'd_ft', 'd_3p']
    
    for attr in attributes:
        made_col = f'{attr}_pct'
        df_copy[made_col] = np.where(df_copy[f'{attr}a'] > 0, df_copy[f'{attr}m'] / df_copy[f'{attr}a'], 0)
    
    df_copy['o_reb_pct'] =  np.where(df_copy['o_reb'] > 0, df_copy['o_oreb'] / df_copy['o_reb'], 0)
    df_copy['d_reb_pct'] =  np.where(df_copy['d_reb'] > 0, df_copy['d_oreb'] / df_copy['d_reb'], 0)
    df_copy.drop(columns = ['o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 'd_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa','d_reb','o_reb','o_oreb','d_oreb','o_dreb','d_dreb'], axis = 1, inplace = True)
    
    return df_copy

def fs_players(df_players):
    df = df_players.copy()
    df.columns = df.columns.str.lower()
    
    stats = [
    'minutes', 'points', 'orebounds', 'drebounds', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'pf',
    'fgattempted', 'fgmade', 'ftattempted', 'ftmade', 'threeattempted', 'threemade', 'dq']
    
    for x in stats:
        soma = ((df[x] * 0.5) + (df['post'+x] * 1.5))
        df[f'total_{x}'] = soma
    
    pct_stats = ["total_fg","total_ft","total_three"]
    
    for attr in pct_stats:
        made_col = f'{attr}_pct'
        df[made_col] = np.where(df[f'{attr}attempted'] > 0, df[f'{attr}made'] / df[f'{attr}attempted'], 0)
        df.drop(columns = [f'{attr}attempted',f'{attr}made'], axis = 1, inplace = True)
    
    df['total_orebounds_pct'] =  np.where(df['total_rebounds'] > 0, df['total_orebounds'] / df['total_rebounds'], 0)
    
    
    df.drop(columns = ['total_orebounds','total_rebounds','total_drebounds'], axis = 1, inplace = True)
    
    
    df.drop(columns = ['minutes', 'points',
    'orebounds', 'drebounds', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'pf',
    'fgattempted', 'fgmade', 'ftattempted', 'ftmade', 'threeattempted', 'threemade', 'dq',
    'postminutes', 'postpoints', 'postorebounds', 'postdrebounds',
    'postrebounds', 'postassists', 'poststeals', 'postblocks', 'postturnovers', 'postpf',
    'postfgattempted', 'postfgmade', 'postftattempted', 'postftmade', 'postthreeattempted',
    'postthreemade', 'postdq','postgs','gs','playerid'], axis = 1, inplace = True)
    
    df = df.groupby(['year', 'tmid']).sum().reset_index()
    
    return df
    

def correlation_matrix(df_corr):
    numeric_df = df_corr.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(20, 16))

    # Create a heatmap
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=.5)
    
    plt.show()

def bisserial_corr(df):
    correlations = {}

    # Iterate over your continuous attributes
    for column in df.columns:
        if(column == 'tmid' or column == "year" or column =="tmID"):
            continue
        
        if (column != 'playoff'):
            correlation, _ = pointbiserialr(df[column], df['playoff'])
            correlations[column] = correlation

    sorted_correlations = dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True))
    

    for feature in sorted_correlations:
        print(f'{feature}: {abs(sorted_correlations[feature]) * 100:.2f}% correlation')
    
    """
    plt.figure(figsize=(20, 16))
    plt.bar(sorted_correlations.keys(), [abs(val) * 100 for val in sorted_correlations.values()])
    plt.xlabel('Features')
    plt.ylabel('Correlation (%)')
    plt.title('Correlation of Features with Playoff')
    plt.show()
    """