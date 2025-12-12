import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def main():
    print("Starting Optuna Search ----------")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    print(" - Best parameters found - ")
    print(study.best_params)
    print(f"Best CV Score: {study.best_value}")


def get_data(filename):
    df = pd.read_csv(filename)

    return df


def objective(trial):
    '''
    Function running several times, to optimize the given function
    '''

    df_train = get_data("data/train.csv")

    cat_features = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

    for col in cat_features:
        df_train[col] = df_train[col].astype(str)

    X = df_train.drop(columns=['diagnosed_diabetes'])
    y = df_train['diagnosed_diabetes']

    # Parameter which are getting optimized later on
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),

        'iterations': 1000,
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'cat_features': cat_features,
        'thread_count': -1
    }

    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[val_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=False
        )

        val_preds = model.predict_proba(X_test)[:, 1]
        scores.append(roc_auc_score(y_test, val_preds))

    return np.mean(scores)


if __name__ == "__main__":
    main()
