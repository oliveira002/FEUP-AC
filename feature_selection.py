import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
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
        