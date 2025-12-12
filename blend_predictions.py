import pandas as pd

def main():
    catboost_submission = pd.read_csv("submissions/2025_12_12_catboost_kfold.csv")
    xgb_submission = pd.read_csv("submissions/2025_12_09_xgb_kfold_0.7231.csv")

    print("calculate new results")
    blend_pred = pd.DataFrame({
        'id': xgb_submission['id'],
        'diagnosed_diabetes': (0.3 * xgb_submission['diagnosed_diabetes']) + (0.7 * catboost_submission['diagnosed_diabetes'])
    })

    print("convert to csv")
    blend_pred.to_csv("submissions/2025_12_12_diabetes_prediction_blend_cat_xgb.csv", index=False)

if __name__ == "__main__":
    main()
