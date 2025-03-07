import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata 
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

train = pd.read_csv("../input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("../input/equity-post-HCT-survival-predictions/test.csv")

RMV = ["ID","efs","efs_time","y","efs_time2"]
FEATURES = [c for c in train.columns if not c in RMV]

CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")

from lifelines import  NelsonAalenFitter
def create_nelson(data):
    data=data.copy()
    naf = NelsonAalenFitter(nelson_aalen_smoothing=0)
    naf.fit(durations=data['efs_time'], event_observed=data['efs'])
    return naf.cumulative_hazard_at_times(data['efs_time']).values*-1
    

combined = pd.concat([train,test],axis=0,ignore_index=True)
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()

from sklearn.model_selection import KFold,StratifiedKFold
from lightgbm import LGBMRegressor
import lightgbm as lgb

train["y_nel"] = create_nelson(train)

#important
train.loc[train.efs == 0, "y_nel"] = (-(-train.loc[train.efs == 0, "y_nel"])**0.5)

FOLDS = 10
def create_stratified_folds(data, target, n_splits=10):
    data['fold'] = -1
    # num_bins = int(np.floor(1 + np.log2(len(data))))  # Sturges' rule for binning
    if (target!="race_group"):
        data['bins'] = pd.qcut(data[target], q=50, duplicates='drop',labels=False)
    data["bins"]=data["race_group"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(data, data['bins'])):
        data.loc[val_idx, 'fold'] = fold
    
    data = data.drop(columns=['bins'])
    return data

train=create_stratified_folds(train,"race_group",FOLDS)
X = train[FEATURES]
y = train["y_nel"]

import optuna
from sklearn.model_selection import KFold
from metric import score

def objective(trial):
    # =======================
    # パラメータのサンプリング
    # =======================
    # 特に重要度の高いものを優先的に調整
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
    max_iter = trial.suggest_int("max_iter", 1000, 2000, step=50)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 48, log=True)
    max_depth = trial.suggest_int("max_depth", 10, 30)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 25, 50)
    l2_regularization = trial.suggest_float("l2_regularization", 1e-8, 1e-2, log=True)
    max_bins = trial.suggest_int("max_bins", 100, 255)
    
    # ===================
    # モデルの定義 (HGBR)
    # ===================
    model = HGBR(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_bins=max_bins,
        # 下記は必要に応じて
        # loss="squared_error",
        # early_stopping=False,
        random_state=42,
        categorical_features=CATS
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    # --- 3. k-fold で学習＆評価 ---
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 学習（early_stopping を使いたい場合は eval_set, early_stopping_rounds を指定）
        model.fit(
            X_train, y_train
        )

        # 予測
        y_pred = train[["ID"]].iloc[val_idx].copy()
        y_pred["prediction"] = model.predict(X_val)

        # 自作スコア算出
        fold_score = score(train[["ID","efs","efs_time","race_group"]].iloc[val_idx].copy(), y_pred, "ID")
        scores.append(fold_score)

    # --- 4. fold の平均スコアを最終評価値として返す ---
    return np.mean(scores)

# ========= Optuna で最適化実行 =========
# 「最大化」を指定（自作スコアが大きいほど良いという扱い）
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # 試行回数=30 (例)

# 0.6699832895289106 and parameters: {'learning_rate': 0.015865608439848605, 'max_iter': 1500, 'max_leaf_nodes': 24, 'max_depth': 16, 'min_samples_leaf': 35, 'l2_regularization': 0.04728867704353522, 'max_bins': 88}. Best is trial 30 with value: 0.6699832895289106.
# 0.6702724830149281 and parameters: {'learning_rate': 0.028100803080866436, 'max_iter': 1650, 'max_leaf_nodes': 36, 'max_depth': 19, 'min_samples_leaf': 23, 'l2_regularization': 0.00016285549301989487, 'max_bins': 151}. Best is trial 22 with value: 0.6702724830149281