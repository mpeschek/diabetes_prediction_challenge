import time
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def main():
    df = get_data("data/train.csv")
    
    X_train, X_test, y_train, y_test = split_data(df)

    # correlation_matrix = get_correlation(X_train, y_train)

    trained_model = gradient_boost_machine_xgb(X_train, X_test, y_train, y_test)

    # Read and process the real test data
    df_test_processed = get_data("data/test.csv")
    test_ids = df_test_processed.index

    # Predict test data with trained model
    final_probabilities = trained_model.predict_proba(df_test_processed)
    submission_values = final_probabilities[:, 1]

    # Create a submission file
    submis = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': submission_values
    })
    submis.to_csv("submissions/2025_12_08_diabetes_prediction_xgboost.csv", index=False)


def get_data(filename):
    '''
    import and clean the data
    '''
    df = pd.read_csv(filename, index_col=0)
    df = pd.get_dummies(
        df,
        columns=["gender", "ethnicity", "education_level", "income_level", "smoking_status", "employment_status"],
        drop_first=False
    )

    # Checked for .isna() but it is 0 in each column
    # print(df.isna().sum())

    # Reorder the columns so the "diagnosed_diabetes" columns is still at the end
    if filename == "data/train.csv":
        reorder_columns = [col for col in df.columns if col != "diagnosed_diabetes"]
        reorder_columns.append("diagnosed_diabetes")
        df = df[reorder_columns]

    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df


def split_data(df):
    '''
    Split the data of the DataFrame to a test and training data
    '''
    X = df.drop(columns=['diagnosed_diabetes'])
    y = df['diagnosed_diabetes']
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        train_size=0.8, 
        random_state=42, 
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def get_correlation(X, y):
    '''
    Get correlation matrix only from the training data
    !!! Not include the test data !!!
    '''
    analysis_df = X.copy()
    analysis_df['diagnosed_diabetes'] = y
    
    corr = analysis_df.corr(method="pearson")
    print(corr['diagnosed_diabetes'].sort_values())

    return corr


def logistic_regression(X_train, X_test, y_train, y_test):
    '''
    Trains a logistic regression model with automatic scaling and imputation
    '''
    trained_model = logistic_regression_training(X_train, y_train)

    y_predicted = trained_model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_predicted)
    print(f"{highlight.bold}Validation ROC-AUC score (train-test-splitting): {score:.5f}{highlight.end}")

    return trained_model


def logistic_regression_training(X, y):
    '''
    logistic regression training
    '''
    lr_model = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(C=1.0, random_state=42, solver='liblinear')
    )

    print("random logistic regression model getting trained\n...")
    time_start = time.perf_counter()

    lr_model.fit(X, y)

    time_end = time.perf_counter()
    print(f"training finished | training time: {(time_end - time_start):.2f} s")

    return lr_model


def random_forest(X_train, X_test, y_train, y_test):
    '''
    use random forest model
    '''

    trained_model = random_forest_training(X_train, y_train)

    # Use test data set and evaluate the result
    y_predicted = trained_model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_predicted)
    print(f"{highlight.bold}Validation ROC-AUC score (train-test-splitting): {score:.5f}{highlight.end}")

    return trained_model


def random_forest_training(X, y):
    '''
    random forest model
    '''
    rf_model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)

    print("random forest model getting trained\n...")
    time_start = time.perf_counter()

    rf_model.fit(X, y)

    time_end = time.perf_counter()
    print(f"training finished | training time: {(time_end - time_start):.2f} s")

    return rf_model


def gradient_boost_machine_xgb(X_train, X_test, y_train, y_test):
    '''
    get xgboost model
    '''
    trained_model = xgb_train(X_train, X_test, y_train, y_test)

    y_predicted = trained_model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_predicted)
    print(f"{highlight.bold}Validation ROC-AUC score (train-test-splitting): {score:.5f}{highlight.end}")

    return trained_model


def xgb_train(X_train, X_test, y_train, y_test):

    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=50
    )
    
    print("random xgboost getting trained\n...")
    time_start = time.perf_counter()

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    time_end = time.perf_counter()
    print(f"training finished | training time: {(time_end - time_start):.2f} s")
    return model


class highlight:
    bold = '\033[1m'
    end = '\033[0m'


if __name__ == "__main__":
    main()
