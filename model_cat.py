import time
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def main():
    df_train = get_data("data/train.csv")
    df_test = get_data("data/test.csv")

    submis_predictions = train_catboost_kfold(df_train, df_test)

    submis = pd.DataFrame({
        'id': df_test['id'],
        'diagnosed_diabetes': submis_predictions
    })

    submis.to_csv("submissions/2025_12_10_catboost_kfold.csv")



def get_data(filename):
    df = pd.read_csv(filename)

    return df


def train_catboost_kfold(df_train, df_test):

    time_start = time.perf_counter()
    print("start catboost training")

    cat_features = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

    for col in cat_features:
        df_train[col] = df_train[col].astype(str)
        df_test[col] = df_test[col].astype(str)

    X = df_train.drop(columns=['diagnosed_diabetes'])
    y = df_train['diagnosed_diabetes']

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_predictions = np.zeros(len(df_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[val_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            eval_metric='AUC',
            random_state=42,
            verbose=200,
            early_stopping_rounds=100,
            cat_features=cat_features,
            thread_count=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True
        )

        val_preds = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, val_preds)
        print(f"Fold {fold+1} AUC: {score:.5f}")

        test_predictions += model.predict_proba(df_test)[:, 1] / 5

    time_end = time.perf_counter()
    print(f"catboost training finished: {(time_end - time_start):.2f} s")

    return test_predictions


if __name__ == "__main__":
    main()
