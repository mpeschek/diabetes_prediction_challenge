import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


def main():
    df_train = get_data("data/train.csv")
    df_test = get_data("data/test.csv")

    submis_values = train_nn(df_train, df_test) 

    submis = pd.DataFrame({
        'id': df_test.index,
        'diagnosed_diabetes': submis_values
    })

    submis.to_csv("submissions/2025_12_12_nn.csv", index=False)


def get_data(filename):
    '''
    import and clean the data
    '''
    df = pd.read_csv(filename, index_col=0)

    df = encode_features(df)
    df = pd.get_dummies(
        df,
        columns=["gender", "ethnicity", "smoking_status", "employment_status"],
        drop_first=True
    )
    # Checked for .isna() but it is 0 in each column
    # print(df.isna().sum())

    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    # df = feature_engineering(df)

    # Reorder the columns so the "diagnosed_diabetes" columns is still at the end
    if filename == "data/train.csv":
        reorder_columns = [col for col in df.columns if col != "diagnosed_diabetes"]
        reorder_columns.append("diagnosed_diabetes")
        df = df[reorder_columns]

    return df


def encode_features(df):

    edu_level = {
        'No formal' : 0,
        'Highschool': 1,
        'Graduate': 2,
        'Postgraduate': 3
    }
    inc_level = {
        'Low': 0,
        'Lower-Middle': 1,
        'Middle': 2,
        'Upper-Middle': 3,
        'High': 4
    }

    df['education_level'] = df['education_level'].map(edu_level)
    df['income_level'] = df['income_level'].map(inc_level)

    return df


def train_nn(df_train, df_test):
    '''
    Training neural network (preprocessed with pandas | not with sklearn)
    '''
    
    X = df_train.drop(columns=['diagnosed_diabetes'])
    y = df_train['diagnosed_diabetes']

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
    )

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros(len(df_test))
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        val_preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold}: {score:.5f}")
        scores.append(score)

        test_preds += model.predict_proba(df_test)[:, 1] / 5

    print(f"Average NN Score: {np.mean(scores)}")

    return test_preds


if __name__ == "__main__":
    main()
