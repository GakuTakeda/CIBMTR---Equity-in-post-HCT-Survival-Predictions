import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata 
import warnings
warnings.filterwarnings('ignore')

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
    # --- 1. 調整したいパラメータを trial からサンプリング ---
    learning_rate = trial.suggest_loguniform('learning_rate', 0.008, 0.015)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 20)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-2, 1e-1, log=True)
    reg_alpha = trial.suggest_float('reg_alpha', 0.015, 0.5, log=True)
    num_leaves = trial.suggest_int('num_leaves', 20, 35)
    max_depth = trial.suggest_int('max_depth', 6, 13)
    num_iterations = trial.suggest_int('num_iterations', 5250, 5650, step=50)
    max_bin = trial.suggest_int('max_bin', 200, 256, step=8)

    # 固定パラメータ（例）
    fixed_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'extra_trees': True,
        'verbose': -1,
        'seed': 42,
        'device': 'cpu',
    }

    # --- 2. クロスバリデーション（5-fold）を用意 ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 評価スコアを格納するリスト
    scores = []

    # --- 3. k-fold で学習＆評価 ---
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMRegressor(
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            num_leaves=num_leaves,
            max_depth=max_depth,
            num_iterations=num_iterations,
            max_bin=max_bin,
            **fixed_params
        )
        # 学習（early_stopping を使いたい場合は eval_set, early_stopping_rounds を指定）
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=0)]
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

 #score: 0.6740467722917443 parameters: {'learning_rate': 0.009453691809675214, 'min_child_samples': 14, 'reg_lambda': 0.038763067060145956, 'reg_alpha': 0.023205959833390973, 'num_leaves': 23, 'max_depth': 7}
 #score: 0.674140767427265  parameters: {'learning_rate': 0.010249196427607903, 'min_child_samples': 7, 'reg_lambda': 0.04381443853306703, 'reg_alpha': 0.010856948539014647, 'num_leaves': 30, 'max_depth': 6, 'num_iterations': 5500}
 #score: 0.6743606622729108 parameters: {'learning_rate': 0.010247473758487297, 'min_child_samples': 14, 'reg_lambda': 0.025968901196413107, 'reg_alpha': 0.010082657579302273, 'num_leaves': 26, 'max_depth': 11, 'num_iterations': 5300, 'max_bin': 224}
 #score: 0.6745001987295776 parameters: {'learning_rate': 0.011280997556651065, 'min_child_samples': 10, 'reg_lambda': 0.021238168657597097, 'reg_alpha': 0.1728826435936071, 'num_leaves': 28, 'max_depth': 13, 'num_iterations': 5500, 'max_bin': 240}
