import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata 

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
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
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

oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i in range(FOLDS):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train.fold!=i,FEATURES].copy()
    y_train = train.loc[train.fold!=i,"y_nel"]
    x_valid = train.loc[train.fold==i,FEATURES].copy()
    y_valid = train.loc[train.fold==i,"y_nel"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        # device="cuda",
        max_depth=4,  
        colsample_bytree=0.55,  
        subsample=0.8,  
        n_estimators=5000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        early_stopping_rounds=200,
        n_jobs=4
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # INFER OOF
    oof_xgb[train.index[train.fold==i]] = (model_xgb.predict(x_valid))
    # INFER TEST
    pred_xgb += (model_xgb.predict(x_test))


oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i in range(FOLDS):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train.fold!=i,FEATURES].copy()
    y_train = train.loc[train.fold!=i,"y_nel"]
    x_valid = train.loc[train.fold==i,FEATURES].copy()
    y_valid = train.loc[train.fold==i,"y_nel"]
    x_test = test[FEATURES].copy()

    model_lgb = LGBMRegressor(
        objective='regression',
        min_child_samples=10,
        num_iterations=5500,
        learning_rate=0.01128,
        extra_trees=True,
        reg_lambda=0.0212,
        reg_alpha=0.173,
        num_leaves=27,
        metric='rmse',
        max_depth=13,
        device='cpu',
        max_bin=240,
        verbose=-1
        )

    model_lgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)]
    )

    # INFER OOF
    oof_lgb[train.index[train.fold==i]] = (model_lgb.predict(x_valid))
    # INFER TEST
    pred_lgb += (model_lgb.predict(x_test))

from scipy.stats import rankdata
from metric import score 

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb*0.9 + oof_lgb*0.1

m = score(y_true.copy(), y_pred.copy(), "ID")

df = pd.DataFrame({"Overall_CV": [m]})

df.to_csv("xgb and lgb ensemble-4.csv", index=False)

