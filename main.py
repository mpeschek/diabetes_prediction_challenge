import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def main():
    df = get_data("data/train.csv")
    
    X_train, X_test, y_train, y_test = split_data(df)

    # correlation_matrix = get_correlation(X_train, y_train)

    random_forest(X_train, X_test, y_train, y_test)


def get_data(filename):
    '''
    import and clean the train data
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


def random_forest(X_train, X_test, y_train, y_test):
    '''
    use random forest model
    '''

    trained_model = random_forest_training(X_train, y_train)

    # Use test data set and evaluate the result
    y_predicted = trained_model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_predicted)
    print(f"Validation ROC-AUC score: {highlight.bold}{score:.5f}{highlight.end}")

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


class highlight:
    bold = '\033[1m'
    end = '\033[0m'


if __name__ == "__main__":
    main()
