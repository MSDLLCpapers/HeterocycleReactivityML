import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.utils import shuffle
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def model_training(data=None, model=None, descriptors=None, split_method=None, ratio=0.8, test_df=None):
    """
    Train a model using different splitting methods and return performance metrics.
    
    Parameters:
    - data: input dataframe containing features and target variable.
    - model: ML model to be trained.
    - descriptors: list of feature to be used for training.
    - split_method: method to split the data ('random', 'time', 'round').
    - ratio: ratio of training to test data.
    - test_df: optional dataframe for test data.
    
    Returns:
    - trained model
    - performance metrics as a dictionary
    - predictions as a dictionary
    """
    start = time.time()
    df = data.copy()
    
    # Split data based on specified method
    if test_df is not None:
        df_train = df
        df_test = test_df
    elif split_method == "random":
        df = shuffle(df, random_state=0)
        split_index = int(ratio * len(df))
        df_train, df_test = df[:split_index], df[split_index:]
    elif split_method == "time":
        df = df.sort_values('Round_Number', ascending=True)
        split_index = int(ratio * len(df))
        df_train, df_test = df[:split_index], df[split_index:]
    elif split_method == "round":
        max_round = df.Round_Number.max()
        df_train = df[df.Round_Number < max_round]
        df_test = df[df.Round_Number == max_round]
    else:
        raise ValueError("Unexpected value for 'split_method': {}".format(split_method))
    
    # Prepare training data
    X_train = df_train[descriptors] if descriptors is not None else df_train.iloc[:, 3:]
    y_train = df_train['Max_LCAP']
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Prepare testing data
    X_test = df_test[descriptors] if descriptors is not None else df_test.iloc[:, 3:]
    y_test = df_test['Max_LCAP']
    X_test = scaler.transform(X_test)

    # Fit model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    preds = {'smiles': df_test['smiles'], 'test': y_test, 'pred': y_pred}

    # Calculate performance metrics
    performance = {
        'pearson r2': pearsonr(y_test, y_pred)[0] ** 2,
        'mae': mean_absolute_error(y_test, y_pred),
        'r2 score': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    end = time.time()
    
    print(f"Total run time is {end-start:.2f} seconds")

    
    return model, performance, preds  # Return trained model, performance dictionary, and predictions



def cv_KFold_metrics(model="random forest", data=None, descriptors=None, nsplits=20):
    """
    Perform k-fold cross-validation and calculate performance metrics.

    Parameters:
    - model: str or estimator object, model name or instance to train.
    - data: pd.DataFrame, dataset containing features and target.
    - descriptors: list, feature names to be used for training.
    - nsplits: int, number of splits for KFold.

    Returns:
    - performance: dict, contains average metrics from cross-validation.
    """
    start = time.time()

    # Copy data to avoid modifying original dataframe
    df = data.copy()
    X, y = df[descriptors], df['Max_LCAP']
    kf = KFold(n_splits=nsplits)

    # Initialize lists to hold metrics for each fold
    metrics = {
        'pearson r2': [],
        'r2 score': [],
        'rmse': [],
        'mae': []
    }

    # K-fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Choose model
        if model == "random forest":
            estimator = RandomForestRegressor()
        elif model == "linear":
            estimator = SGDRegressor()
        else:
            estimator = model  # If a model instance is provided

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Fit model
        estimator.fit(X_train, y_train)

        # Predictions
        y_pred = estimator.predict(X_test)

        # Calculate metrics
        metrics['pearson r2'].append(pearsonr(y_test, y_pred)[0]**2)
        metrics['r2 score'].append(r2_score(y_test, y_pred))
        metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))

    # Calculate average metrics
    performance = {key: np.mean(value) for key, value in metrics.items()}
    end = time.time()
    print(f"Total run time is {end - start:.2f} seconds")

    return performance



def cv_random_split_metrics(model='random forest', data=None, descriptors=None, nsplits=20, test_size=0.2, metric_method='average performance'):
    """
    Perform random split cross-validation and calculate performance metrics.

    Parameters:
    - model: str or estimator object, model name or instance to train.
    - data: pd.DataFrame, dataset containing features and target.
    - descriptors: list, features to be used for training.
    - nsplits: int, number of random splits for evaluation.
    - test_size: float, proportion of dataset to include in test split.
    - metric_method: str, method for calculating performance metrics.

    Returns:
    - performance: dict, containing average metrics from cross-validation.
    """
    start = time.time()
    pearson_r2s = []
    r2s = []
    rmses = []
    maes = []
    
    df = data.copy()
    X, y = df[descriptors + ['smiles']], df['Max_LCAP']
    y_df = pd.DataFrame(columns=['Molecule_name', 'RF_predicted', 'observed'])

    for i in range(nsplits):
        # Choose the model
        if model == 'random forest':
            estimator = RandomForestRegressor()
        else:
            estimator = model  # If a model instance is provided

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[descriptors])
        X_test_scaled = scaler.transform(X_test[descriptors])
        
        # Fit model
        estimator.fit(X_train_scaled, y_train)
        y_pred = estimator.predict(X_test_scaled)

        # Create dataframe for predictions
        pred_df = pd.DataFrame({
            'Molecule_name': X_test['smiles'],
            'RF_predicted': y_pred,
            'observed': y_test
        })
        
        # Calculate metrics
        pearson_r2s.append(pearsonr(y_test, y_pred)[0]**2)
        r2s.append(r2_score(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        maes.append(mean_absolute_error(y_test, y_pred))
        y_df = pd.concat([pred_df, y_df], ignore_index=True)

    # Group by molecule to calculate means and standard deviations
    y_df['observed'] = y_df['observed'].astype(float)
    df_mean = y_df.groupby('Molecule_name').agg('mean')
    
    performance = {}
    
    # Calculate performance metrics based on specified method
    if metric_method == 'average prediction':
        performance['pearson r2'] = pearsonr(df_mean['observed'], df_mean['RF_predicted'])[0] ** 2
        performance['mae'] = mean_absolute_error(df_mean['observed'], df_mean['RF_predicted'])
        performance['r2 score'] = r2_score(df_mean['observed'], df_mean['RF_predicted'])
        performance['rmse'] = np.sqrt(mean_squared_error(df_mean['observed'], df_mean['RF_predicted']))
        
    elif metric_method == 'average performance':
        performance['pearson r2'] = np.mean(pearson_r2s)
        performance['mae'] = np.mean(maes)
        performance['r2 score'] = np.mean(r2s)
        performance['rmse'] = np.mean(rmses)

    end = time.time()
    print(f"Total run time is {end - start:.2f} seconds")
    
    return performance
