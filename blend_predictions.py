import pandas as pd

def num_blend():
    catboost_submission = pd.read_csv("submissions/2025_12_12_catboost_kfold.csv")
    nn_submission = pd.read_csv("submissions/2025_12_12_nn.csv")

    blend_pred = pd.DataFrame({
        'id': nn_submission['id'],
        'diagnosed_diabetes': (0.15 * nn_submission['diagnosed_diabetes']) + (0.85 * catboost_submission['diagnosed_diabetes'])
    })

    blend_pred.to_csv("submissions/2025_12_12_diabetes_prediction_blend_cat_nn.csv", index=False)


def rank_blend():
    catboost_submission = pd.read_csv("submissions/2025_12_12_catboost_kfold.csv")
    nn_submission = pd.read_csv("submissions/2025_12_12_nn.csv")

    cat_ranks = catboost_submission['diagnosed_diabetes'].rank(pct=True)
    nn_ranks = nn_submission['diagnosed_diabetes'].rank(pct=True)

    blend_ranks = (0.9 * cat_ranks) + (0.1 * nn_ranks)

    submission = pd.DataFrame({
        'id': nn_submission['id'],
        'diagnosed_diabetes': blend_ranks
    })
    submission.to_csv("submissions/2025_12_12_rank_blend_90_10.csv", index=False)


if __name__ == "__main__":
    rank_blend()
