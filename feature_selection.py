import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pointbiserialr
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score




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

def fs_players(df_players, po_weight):
    df = df_players.copy()
    df.columns = df.columns.str.lower()
    df = df.groupby(['year', 'tmid']).sum().reset_index()

    stats = [
    'minutes', 'points', 'orebounds', 'drebounds', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'pf',
    'fgattempted', 'fgmade', 'ftattempted', 'ftmade', 'threeattempted', 'threemade', 'dq','gs','gp']
    
    for x in stats:
        soma = ((df[x] * (1 - po_weight)) + (df['post'+x] * po_weight))
        df[f'total_{x}'] = soma
    
    pct_stats = ["total_fg","total_ft","total_three"]
    
    for attr in pct_stats:
        made_col = f'{attr}_pct'
        df[made_col] = np.where(df[f'{attr}attempted'] > 0, df[f'{attr}made'] / df[f'{attr}attempted'], 0)
        df.drop(columns = [f'{attr}attempted',f'{attr}made'], axis = 1, inplace = True)
    
    df['total_orebounds_pct'] =  np.where(df['total_rebounds'] > 0, df['total_orebounds'] / df['total_rebounds'], 0)
    df['total_drebounds_pct'] =  np.where(df['total_rebounds'] > 0, df['total_drebounds'] / df['total_rebounds'], 0)
    
    df.drop(columns = ['total_orebounds','total_rebounds','total_drebounds'], axis = 1, inplace = True)
    
    
    df.drop(columns = ['minutes', 'points',
    'orebounds', 'drebounds', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'pf',
    'fgattempted', 'fgmade', 'ftattempted', 'ftmade', 'threeattempted', 'threemade', 'dq',
    'postminutes', 'postpoints', 'postorebounds', 'postdrebounds',
    'postrebounds', 'postassists', 'poststeals', 'postblocks', 'postturnovers', 'postpf',
    'postfgattempted', 'postfgmade', 'postftattempted', 'postftmade', 'postthreeattempted',
    'postthreemade', 'postdq','playerid','gs','gp','postgp','postgs'], axis = 1, inplace = True)
        
    return df
    
def correlation_matrix(df_corr, corr):
    numeric_df = df_corr.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(20, 16))

    # Create a triangular mask
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create a heatmap with the triangular mask
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=.5, mask=mask)
    
    plt.show()
    
    # Find pairs with high correlation (> 0.7)
    high_corr_pairs = [(col1, col2) for col1 in corr_matrix.columns for col2 in corr_matrix.columns if col1 < col2 and abs(corr_matrix.loc[col1, col2]) > corr]
    
    # Print pairs with high correlation
    if high_corr_pairs:
        print(f"\n\033[1mPairs with high correlation (> {corr}):\033[0m")
        for col1, col2 in high_corr_pairs:
            print(f"{col1} and {col2}: {corr_matrix.loc[col1, col2]:.2f}")
    else:
        print(f"\nNo pairs with high correlation (> {corr}) found.")

def bisserial_corr(df):
    correlations = {}

    # Iterate over your continuous attributes
    for column in df.columns:
        if(column == 'tmid' or column == "year" or column =="tmID" or column == "confID" or column == "confid"):
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

def grid_search(features,x_train,x_test,y_train,y_test):
    
    param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
    }

    param_grid_lr = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['liblinear', 'saga'],
    'max_iter': [10000]
    }

    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    param_grid_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    # Create instances of the models
    rf = RandomForestClassifier(random_state=42)
    lr = LogisticRegression(random_state=42)
    svm = SVC(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    knn = KNeighborsClassifier()

    # Create a dictionary of models and their corresponding parameter grids
    models = {'Random Forest': (rf, param_grid_rf),
            'Logistic Regression': (lr, param_grid_lr),
            #'Support Vector Machine': (svm, param_grid_svm),
            'Gradient Boosting': (gb, param_grid_gb),
            'K-Nearest Neighbors': (knn, param_grid_knn)}

    
    dic = {}
    for model_name, (model, param_grid) in models.items():
        x_test_features = x_test[features[model_name]]
        x_train_features = x_train[features[model_name]]
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(x_train_features, y_train)
        
        dic[model_name] = grid_search.best_params_
        
        best_model = grid_search.best_estimator_
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    
    return dic

def plot_cfmatrix_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 5))
    ax1 = plt.subplot(121)
    sb.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=True)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    classification_rep = classification_report(y_true, y_pred)
    ax2 = plt.subplot(122)
    ax2.text(0.05, 0.5, classification_rep, fontsize=12, ha='left')
    ax2.axis('off')

    total_samples = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            percentage = value / total_samples * 100
            ax1.text(j + 0.5, i + 0.35, f'{percentage:.2f}%', ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()
    

def plot_auc_curve(model,x_test,y_test):
    y_pred_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('(ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()