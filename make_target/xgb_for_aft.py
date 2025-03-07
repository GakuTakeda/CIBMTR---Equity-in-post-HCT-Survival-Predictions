import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings('ignore')
from metric import score

test = pd.read_csv("../input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("../input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()
combined = pd.concat([train,test],axis=0,ignore_index=True)
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
RMV = ["ID","efs","efs_time","kap", "aln"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")
combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()
from sklearn.model_selection import KFold,StratifiedKFold
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
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train_time = train.loc[train_index,"efs_time"]
    y_train_status = train.loc[train_index,"efs"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid_time = train.loc[test_index,"efs_time"]
    y_valid_status = train.loc[test_index,"efs"]
    x_test = test[FEATURES].copy()
    print(x_train.shape)
    y_train_lower = y_train_time.copy()
    y_train_upper = np.where(y_train_status == 1, y_train_time, 1e10)
    y_train_aft = np.vstack([y_train_lower, y_train_upper]).T # shape: (N, 2)

    y_valid_lower = y_valid_time.copy()
    y_valid_upper = np.where(y_valid_status == 1, y_valid_time, 1e10)
    y_valid_aft = np.vstack([y_valid_lower, y_valid_upper]).T # shape: (M, 2)

    # ---- DMatrix作成 (カテゴリを扱う場合は enable_categorical=True) ----
    dtrain = xgb.DMatrix(data=x_train, label=y_train_aft, enable_categorical=True)
    dvalid = xgb.DMatrix(data=x_valid, label=y_valid_aft, enable_categorical=True)
    dtest = xgb.DMatrix(data=x_test, enable_categorical=True)

    # ---- ハイパーパラメータ設定 ----
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        #"tree_method": "gpu_hist",  # GPUが使えない/不安定なら "hist" に切り替え
        "device": "cuda",           # GPU使用; CPUなら指定不要
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "learning_rate": 0.02,
        "min_child_weight": 80,
        "reg_alpha": 0.55,
        "reg_lambda": 6.78,
    }

    # ---- 学習 (booster API) ----
    # evals に (dvalid, "valid") を渡すと検証用スコアが出力される
    model_xgb = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=200,
        evals=[(dvalid, "valid")],
        verbose_eval=50,
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
xgb_for_aft = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost AFT =",xgb_for_aft )