from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle

def opt_hyper_param(data=None, descriptors=None, split_method='random', ratio=0.8):
    """
    Tune hyperparameters of a Random Forest model using RandomizedSearchCV.

    Parameters:
    - data: pd.DataFrame, the dataset containing features and target values.
    - descriptors: list, features to be used for training.
    - split_method: str, method to split the data ('random', 'time', 'round').
    - ratio: float, proportion of data to use for training/validation.

    Returns:
    - best_estimator: the best fitted Random Forest model.
    - df_train_val: DataFrame containing the training/validation data.
    - df_test: DataFrame containing the testing data.
    """
    
    # Copy input data to avoid modifying original dataframe
    df = data.copy()

    # Split data based on selected method
    if split_method == 'random':
        df = shuffle(df, random_state=115)
        split_index = int(ratio * len(df))
        df_train_val, df_test = df[:split_index], df[split_index:]
    elif split_method == "time":
        df = df.sort_values('Round_Number', ascending=True)
        split_index = int(ratio * len(df))
        df_train_val, df_test = df[:split_index], df[split_index:]
    elif split_method == "round":
        max_round = df.Round_Number.max()
        df_train_val = df[df.Round_Number < max_round]
        df_test = df[df.Round_Number == max_round]
    else:
        raise ValueError(f"Unexpected value of 'split_method': {split_method}")

    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Initialize Random Forest regressor and RandomizedSearchCV
    rf = RandomForestRegressor(random_state=115)
    random_search = RandomizedSearchCV(rf, param_grid, n_iter=100, cv=10, random_state=115, n_jobs=-1)

    # Prepare data for training
    X = df_train_val[descriptors]
    y = df_train_val['Max_LCAP']
    
    # Fit RandomizedSearchCV
    random_search.fit(X, y)

    # Return best estimator and train/test dataframes
    return random_search.best_estimator_, df_train_val, df_test
