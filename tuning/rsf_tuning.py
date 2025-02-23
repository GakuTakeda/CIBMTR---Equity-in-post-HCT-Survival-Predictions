#pip install scikit-survival
import optuna
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import pandas as pd
pd.options.display.max_columns = None
from scipy.stats import rankdata 
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import tqdm
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
import csv



class CFG:

    train_path = Path('../input/equity-post-HCT-survival-predictions/train.csv')
    test_path = Path('../input/equity-post-HCT-survival-predictions/test.csv')
    subm_path = Path('../input/equity-post-HCT-survival-predictions/sample_submission.csv')

    batch_size = 32768
    early_stop = 50
    penalizer = 0.01
    n_splits = 5

class FE:

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def _load_data(self, path):

        return pl.read_csv(path, batch_size=self._batch_size)

    def _update_hla_columns(self, df):
        
        df = df.with_columns(
            
            pl.col('hla_match_a_low').fill_null(0)
            .add(pl.col('hla_match_b_low').fill_null(0))
            .add(pl.col('hla_match_drb1_high').fill_null(0))
            .alias('hla_nmdp_6'),
            
            pl.col('hla_match_a_low').fill_null(0)
            .add(pl.col('hla_match_b_low').fill_null(0))
            .add(pl.col('hla_match_drb1_low').fill_null(0))
            .alias('hla_low_res_6'),
            
            pl.col('hla_match_a_high').fill_null(0)
            .add(pl.col('hla_match_b_high').fill_null(0))
            .add(pl.col('hla_match_drb1_high').fill_null(0))
            .alias('hla_high_res_6'),
            
            pl.col('hla_match_a_low').fill_null(0)
            .add(pl.col('hla_match_b_low').fill_null(0))
            .add(pl.col('hla_match_c_low').fill_null(0))
            .add(pl.col('hla_match_drb1_low').fill_null(0))
            .alias('hla_low_res_8'),
            
            pl.col('hla_match_a_high').fill_null(0)
            .add(pl.col('hla_match_b_high').fill_null(0))
            .add(pl.col('hla_match_c_high').fill_null(0))
            .add(pl.col('hla_match_drb1_high').fill_null(0))
            .alias('hla_high_res_8'),
            
            pl.col('hla_match_a_low').fill_null(0)
            .add(pl.col('hla_match_b_low').fill_null(0))
            .add(pl.col('hla_match_c_low').fill_null(0))
            .add(pl.col('hla_match_drb1_low').fill_null(0))
            .add(pl.col('hla_match_dqb1_low').fill_null(0))
            .alias('hla_low_res_10'),
            
            pl.col('hla_match_a_high').fill_null(0)
            .add(pl.col('hla_match_b_high').fill_null(0))
            .add(pl.col('hla_match_c_high').fill_null(0))
            .add(pl.col('hla_match_drb1_high').fill_null(0))
            .add(pl.col('hla_match_dqb1_high').fill_null(0))
            .alias('hla_high_res_10'),
            
        )
    
        return df

    def _cast_datatypes(self, df):

        num_cols = [
            'hla_high_res_8',
            'hla_low_res_8',
            'hla_high_res_6',
            'hla_low_res_6',
            'hla_high_res_10',
            'hla_low_res_10',
            'hla_match_dqb1_high',
            'hla_match_dqb1_low',
            'hla_match_drb1_high',
            'hla_match_drb1_low',
            'hla_nmdp_6',
            'year_hct',
            'hla_match_a_high',
            'hla_match_a_low',
            'hla_match_b_high',
            'hla_match_b_low',
            'hla_match_c_high',
            'hla_match_c_low',
            'donor_age',
            'age_at_hct',
            'comorbidity_score',
            'karnofsky_score',
            'efs',
            'efs_time'
        ]

        for col in df.columns:

            if col in num_cols:
                df = df.with_columns(pl.col(col).fill_null(-1).cast(pl.Float32))  

            else:
                df = df.with_columns(pl.col(col).fill_null('Unknown').cast(pl.String))  

        return df.with_columns(pl.col('ID').cast(pl.Int32))


    def apply_fe(self, path):

        df = self._load_data(path)   
        df = self._update_hla_columns(df)                     
        df = self._cast_datatypes(df)        
        df = df.to_pandas()
        
        cat_cols = [col for col in df.columns if df[col].dtype == pl.String]

        return df, cat_cols

fe = FE(CFG.batch_size)

train_data, cat_cols = fe.apply_fe(CFG.train_path)

X = train_data.drop(columns=['efs', 'efs_time', 'ID'])
y = train_data[['efs', 'efs_time']]
y = Surv.from_arrays(event=y['efs'], time=y['efs_time'])
def get_dummy(data, columns=cat_cols):
  return pd.get_dummies(data, columns=cat_cols, drop_first=True)
X = get_dummy(X)
def objective(trial):
    # Optuna によるハイパーパラメータの探索範囲
    n_estimators = trial.suggest_int("n_estimators", 50, 150, step=10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_depth = trial.suggest_int("max_depth", 3, 20)

    # RSF モデルの定義
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=42
    )
    KFold_ = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=42)
    c_index_list = []
    for train_index, test_index in KFold_.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rsf.fit(X_train, y_train)
        c_index = concordance_index_censored(y_test["event"], y_test["time"], rsf.predict(X_test))[0]
        c_index_list.append(c_index)

    return np.mean(c_index_list)


# 5. Optuna で最適化を実行
study = optuna.create_study(direction="maximize")  # C-index を最大化
study.optimize(objective, n_trials=50)  # 試行回数を 50 に設定

# 6. 最適なハイパーパラメータを取得
best_params = study.best_trial.params
best_score = study.best_trial.value

# CSVファイルのパスを設定（現在のフォルダに保存）
csv_file_path = "best_trial_rsf.csv"

# 7. CSV ファイルに保存
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # ヘッダー行を追加
    writer.writerow(["Parameter", "Value"])
    
    # パラメータを書き込み
    for key, value in best_params.items():
        writer.writerow([key, value])
    
    # C-index も追加
    writer.writerow(["Best C-index", best_score])

print(f"Best trial results saved to {csv_file_path}")
