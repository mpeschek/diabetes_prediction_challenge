import pandas as pd

def main():
    lr_submission = pd.read_csv("submissions/2025_12_08_diabetes_prediction_logistic_regression.csv")
    xgb_submission = pd.read_csv("submissions/2025_12_09_xgb_kfold_0.7231.csv")

    blend_pred = pd.DataFrame({
        'id': xgb_submission['id'],
        'diagnosed_diabetes': (0.7 * xgb_submission['diagnosed_diabetes']) + (0.3 * lr_submission['diagnosed_diabetes'])
    })

    blend_pred.to_csv("submissions/2025_12_09_diabetes_prediction_blend_lr_xgb.csv", index=False)

if __name__ == "__main__":
    main()
