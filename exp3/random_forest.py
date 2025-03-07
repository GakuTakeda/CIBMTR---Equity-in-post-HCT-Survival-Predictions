import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata 
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.impute import SimpleImputer, KNNImputer

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
imputer = KNNImputer(n_neighbors=5)
def get_dummy(data, columns=CATS):
  return pd.get_dummies(data, columns=columns)
X = get_dummy(X)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

def objective(trial):
    # =====================
    #  1. ハイパラをサンプリング
    # =====================
    n_estimators = trial.suggest_int("n_estimators", 100, 500, step=10)
    max_depth = trial.suggest_int("max_depth", 20, 40)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])  # Noneは全説明変数
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 70, 256, log=True)
    min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.1)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.001)
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.001)
    # 0.6258041151128194 and parameters: {'n_estimators': 200, 'max_depth': 28, 'max_features': 'log2', 'bootstrap': False, 'max_leaf_nodes': 124, 'min_weight_fraction_leaf': 0.028305422128100777, 'min_impurity_decrease': 0.0003408432934447886, 'ccp_alpha': 0.00032687722476803404}
#0.6566553958478558.{'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 7, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'bootstrap': False, 'max_leaf_nodes': 84, 'min_weight_fraction_leaf': 0.18339115885496424, 'min_impurity_decrease': 0.00026095296642912255, 'min_impurity_split': 0.27752486516167274, 'ccp_alpha': 0.0002038254511892803, }. 
    # =====================
    #  2. モデルの定義
    # =====================
    model = RFR(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=False,
        max_leaf_nodes=max_leaf_nodes,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha,
        random_state=42,
        n_jobs=-1,  # CPUコアを最大限使う
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
study.optimize(objective, n_trials=50)  

#{'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 7, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'bootstrap': False, 'max_leaf_nodes': 84, 'min_weight_fraction_leaf': 0.18339115885496424, 'min_impurity_decrease': 0.00026095296642912255, 'min_impurity_split': 0.27752486516167274, 'ccp_alpha': 0.0002038254511892803, 'max_samples': 0.6318967225731346}. Best is trial 13 with value: 0.6566553958478558.