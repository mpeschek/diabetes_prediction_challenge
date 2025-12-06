import pandas as pd


def main():
    df = get_data()


def get_data():
    df = pd.read_csv("data/train.csv", index_col=0)
    df = pd.get_dummies(df, columns=["gender", "ethnicity", "education_level", "income_level", "smoking_status", "employment_status"], drop_first=False)

    # Checked for .isna() but it is 0 in each column
    print(df.isna().sum())

    # Reorder the columns so the "diagnosed_diabetes" columns is still at the end
    reorder_columns = [col for col in df.columns if col != "diagnosed_diabetes"]
    reorder_columns.append("diagnosed_diabetes")
    df = df[reorder_columns]

    return df


if __name__ == "__main__":
    main()
