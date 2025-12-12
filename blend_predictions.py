import pandas as pd

def main():
    catboost_submission = pd.read_csv("submissions/2025_12_12_catboost_kfold.csv")
    nn_submission = pd.read_csv("submissions/2025_12_12_nn.csv")

    print("calculate new results")
    blend_pred = pd.DataFrame({
        'id': nn_submission['id'],
        'diagnosed_diabetes': (0.15 * nn_submission['diagnosed_diabetes']) + (0.85 * catboost_submission['diagnosed_diabetes'])
    })

    print("convert to csv")
    blend_pred.to_csv("submissions/2025_12_12_diabetes_prediction_blend_cat_nn.csv", index=False)

if __name__ == "__main__":
    main()
