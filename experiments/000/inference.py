import pickle
import warnings

import config  # edit config.py as needed
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from lifelines import KaplanMeierFitter
from scipy.stats import rankdata
from seed import seed_everything  # edit seed.py as needed
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")


# ====================================================
# Configurations
# ====================================================
class CFG:
    DYR_RUN = False
    EXP_NAME = config.EXP_NAME
    AUTHOR = "marumarukun"
    COMPETITION = config.KAGGLE_COMPETITION_NAME
    DATA_PATH = config.COMP_DATASET_DIR
    OUTPUT_DIR = config.OUTPUT_DIR
    MODEL_PATH = config.ARTIFACT_EXP_DIR(config.EXP_NAME) / "models"
    METHOD_LIST = ["lightgbm", "xgboost", "catboost"]
    SEED = 42
    n_folds = 2 if DYR_RUN else 10
    target_col_list = ["y"]
    # group_col = "race_group"  # Required for GroupKFold (edit as needed)
    # stratified_col = "race_group"  # Required for StratifiedKFold (edit as needed)
    stratified_cols = ["efs", "race_group"]  # Required for MultilabelStratifiedKFold（edit as needed）
    num_boost_round = 100 if DYR_RUN else 1000000
    early_stopping_round = 10 if DYR_RUN else 100
    verbose = 500

    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    regression_lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.02,
        # "num_leaves": 31,
        "max_depth": 3,
        "feature_fraction": 0.8,
        "seed": SEED,
        "device": "cuda",  # cpu/gpu/cuda
    }
    # https://xgboost.readthedocs.io/en/stable/parameter.html
    regression_xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "learning_rate": 0.02,
        "max_depth": 3,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "enable_categorical": True,
        "min_child_weight": 80,
        "random_state": SEED,
        "device": "cuda",  # cpu/gpu/cuda
    }
    # https://catboost.ai/docs/en/references/training-parameters/
    regression_cat_params = {
        "loss_function": "RMSE",
        "learning_rate": 0.02,
        "iterations": num_boost_round,
        # "depth": 7,
        "grow_policy": "Lossguide",
        "random_seed": SEED,
        "task_type": "GPU",  # CPU/GPU
    }

    model_weight_dict = {"lightgbm": 0.40, "xgboost": 0.30, "catboost": 0.30}


# ====================================================
# Seed everything
# ====================================================
seed_everything(CFG.SEED)


# ====================================================
# Read data
# ====================================================
train = pl.read_csv(CFG.DATA_PATH / "train.csv", try_parse_dates=True)
test = pl.read_csv(CFG.DATA_PATH / "test.csv", try_parse_dates=True)


# ====================================================
# Preprocess(ここに前処理や特徴量エンジニアリングを記述)
# ====================================================
def transform_survival_probability(df, time_col="efs_time", event_col="efs"):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y


y = transform_survival_probability(train, time_col="efs_time", event_col="efs")
train = train.with_columns(pl.Series(y).alias("y"))


train = train.to_pandas()
test = test.to_pandas()

# ====================================================
# Set categorical columns etc. (pandas operation from here)
# ====================================================
RMV = ["ID", "efs", "efs_time", "y", "fold"]
FEATURES = [c for c in train.columns if c not in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")

CATS = []
for c in FEATURES:
    if train[c].dtype == "object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")

combined = pd.concat([train, test], axis=0, ignore_index=True)
# print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ", end="")
for c in FEATURES:
    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ", end="")
        combined[c], _ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")

    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype == "float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype == "int64":
            combined[c] = combined[c].astype("int32")

train = combined.iloc[: len(train)].copy()
test = combined.iloc[len(train) :].reset_index(drop=True).copy()


# ====================================================
# Inference functions
# ====================================================
def lightgbm_inference(x_test: pd.DataFrame, target_col: str):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model = pickle.load(
            open(
                CFG.MODEL_PATH / f"lightgbm_{target_col}_fold{fold + 1}_seed{CFG.SEED}_ver{CFG.EXP_NAME}.pkl",
                "rb",
            )
        )
        # Predict
        pred = model.predict(x_test)
        test_pred += pred
    return test_pred / CFG.n_folds


def xgboost_inference(x_test: pd.DataFrame, target_col: str):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model = pickle.load(
            open(
                CFG.MODEL_PATH / f"xgboost_{target_col}_fold{fold + 1}_seed{CFG.SEED}_ver{CFG.EXP_NAME}.pkl",
                "rb",
            )
        )
        # Predict
        pred = model.predict(xgb.DMatrix(x_test, enable_categorical=True))
        test_pred += pred
    return test_pred / CFG.n_folds


def catboost_inference(x_test: pd.DataFrame, target_col: str):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model = pickle.load(
            open(
                CFG.MODEL_PATH / f"catboost_{target_col}_fold{fold + 1}_seed{CFG.SEED}_ver{CFG.EXP_NAME}.pkl",
                "rb",
            )
        )
        # Predict
        pred = model.predict(x_test)
        test_pred += pred
    return test_pred / CFG.n_folds


def gradient_boosting_model_inference(method: str, test_df: pd.DataFrame, features: list, target_col: str):
    x_test = test_df[features]
    if method == "lightgbm":
        test_pred = lightgbm_inference(x_test, target_col)
    if method == "xgboost":
        test_pred = xgboost_inference(x_test, target_col)
    if method == "catboost":
        test_pred = catboost_inference(x_test, target_col)
    return test_pred


def predicting(input_df: pd.DataFrame, features: list):
    output_df = input_df.copy()
    for target_col in CFG.target_col_list:
        # output_df[target_col] = 0
        for method in CFG.METHOD_LIST:
            output_df[f"{method}_pred_{target_col}"] = gradient_boosting_model_inference(
                method, input_df, features, target_col
            )
            # output_df[target_col] += CFG.model_weight_dict[method] * output_df[f"{method}_pred_{target_col}"]
    return output_df


# ====================================================
# Inference
# ====================================================
output_df = predicting(test, FEATURES)
pred_lgb = output_df["lightgbm_pred_y"]
pred_xgb = output_df["xgboost_pred_y"]
pred_cat = output_df["catboost_pred_y"]

submission = pd.read_csv(CFG.DATA_PATH / "sample_submission.csv")
submission["prediction"] = rankdata(pred_lgb) + rankdata(pred_xgb) + rankdata(pred_cat)
submission.to_csv(CFG.OUTPUT_DIR / "submission.csv", index=False)
print("Sub shape:", submission.shape)
submission.head()


# import config
# import polars as pl

# df = pl.read_csv(config.COMP_DATASET_DIR / "sample_submission.csv")
# df.write_csv(config.OUTPUT_DIR / "submission.csv")

# print(config.OUTPUT_DIR)
# print(pl.read_csv(config.OUTPUT_DIR / "submission.csv").shape)

# print(config.ARTIFACT_DIR)
# print(pl.read_csv(config.ARTIFACT_EXP_DIR(config.EXP_NAME) / "submission.csv").shape)

# if not config.IS_KAGGLE_ENV:
#     from src.kaggle_utils.customhub import dataset_upload, model_upload

#     model_upload(
#         handle=config.ARTIFACTS_HANDLE,
#         local_model_dir=config.OUTPUT_DIR,
#         update=False,
#     )
#     dataset_upload(
#         handle=config.CODES_HANDLE,
#         local_dataset_dir=config.ROOT_DIR,
#         update=True,
#     )
