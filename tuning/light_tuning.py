import optuna
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
from lifelines import KaplanMeierFitter


def objective(trial, train, FEATURES, y):
    FOLDS = 5
    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(train))
    pos_shift = 0.2
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 20, 60),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.3, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        "random_state": 42
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 評価スコアを格納するリスト
    scores = []
    

    # --- 3. k-fold で学習＆評価 ---
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=0)]
        )
        oof_lgb[test_index] = model_lgb.predict(x_valid)
    
        y_pred = train[["ID"]].iloc[val_idx].copy()
        y_pred["prediction"] = model.predict(X_val)

        # 自作スコア算出
        fold_score = score(train[["ID","efs","efs_time","race_group"]].iloc[val_idx].copy(), y_pred, "ID")
        scores.append(fold_score)

    # --- 4. fold の平均スコアを最終評価値として返す ---
    return np.mean(scores)
def tune_lgb(train, FEATURES, y, n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train, FEATURES, y), n_trials=n_trials)
    return study.best_params

def add_features(df):
    sex_match = df.sex_match.astype(str)
    sex_match = sex_match.str.split("-").str[0] == sex_match.str.split("-").str[1]
    df['sex_match_bool'] = sex_match.astype("object")
    return df

def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    df_ = df.loc[(df[time_col] < 24) | (df[event_col] == 0)]
    kmf.fit(df_[time_col], df_[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y

from metric import score as score_f

def _prepare_training_data(FEATURES, test, test_index, train, train_index, y, pos_shift=0.1):
    y[train.efs == 0] = y[train.efs == 1].min() - pos_shift
    std = np.std(y[train_index])
    x_train = train.loc[train_index, FEATURES].copy()
    y_train = y[train_index] / std
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = y[test_index] / std
    x_test = test[FEATURES].copy()

    le = LabelEncoder()
    val = train.loc[test_index]
    return x_test, x_train, x_valid, y_train, y_valid

def prepare_data(eps=2e-2, eps_mul=1.1):
    test = pd.read_csv("../input/equity-post-HCT-survival-predictions/test.csv")
    test = add_features(test)
    train = pd.read_csv("../input/equity-post-HCT-survival-predictions/train.csv")
    train = add_features(train)
    train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')
    RMV = ["ID", "efs", "efs_time", "y"]
    FEATURES = [c for c in train.columns if not c in RMV]

    CATS = []
    for c in FEATURES:
        if train[c].dtype == "object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c] = test[c].fillna("NAN")
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    for c in FEATURES:

        if c in CATS:
            combined[c], _ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")

        else:
            if combined[c].dtype == "float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype == "int64":
                combined[c] = combined[c].astype("int32")
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].reset_index(drop=True).copy()
    y = train.y.copy()
    y = (y - y.min() + eps) / (y.max() - y.min() + eps_mul * eps)
    y = np.log(y / (1 - y))
    return FEATURES, test, train, y

def make_predictions_lgb(p):
    hparams = dict(
        eps=2e-2,
        eps_mul=1.01,
        pos_shift=0.2
    )
    FEATURES, test, train, y = prepare_data(hparams.pop('eps'), hparams.pop('eps_mul'))
    best_params = tune_lgb(train, FEATURES, y)
    FOLDS = 5
    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(train))
    pred_lgb = np.zeros(len(test))
    pos_shift = hparams.pop('pos_shift')
    
    for i, (train_index, test_index) in enumerate(kf.split(train, train.race_group)):
        print("#" * 25)
        print(f"### Fold {i + 1}")
        print("#" * 25)
        
        x_test, x_train, x_valid, y_train, y_valid = _prepare_training_data(
            FEATURES, test.copy(), test_index, train.copy(), train_index, y.copy(), pos_shift=pos_shift
        )
        
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)
        
        model_lgb = lgb.train(
            best_params,
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=3000,
            early_stopping_rounds=100,
            verbose_eval=False
        )
        
        oof_lgb[test_index] = model_lgb.predict(x_valid)
        pred_lgb += model_lgb.predict(x_test) / FOLDS
    
    return FOLDS, oof_lgb, pred_lgb, train


def run_lgb(p):
    FOLDS, oof_lgb, pred_lgb, train = make_predictions_lgb(p)
    y_true = train[["ID", "efs", "efs_time", "race_group"]].copy()
    y_pred = train[["ID"]].copy()
    y_pred["prediction"] = oof_lgb
    m = score_f(train.copy(), y_pred.copy(), "ID")
    print(f"\nOverall CV for LightGBM KaplanMeier =", m)
    return pred_lgb, oof_lgb

p = {'n_jobs': -1}
pred_xgb_1, oof_xgb_1 = run_lgb(p) # CV : 0.6821388742966566