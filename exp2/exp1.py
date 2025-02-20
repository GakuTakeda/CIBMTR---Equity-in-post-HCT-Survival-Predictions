


# %%
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'iframe'
pd.options.display.max_columns = None
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
import lightgbm as lgb
import xgboost as xgb
from metric import score
from scipy.stats import rankdata 
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class CFG:

    train_path = Path('../input/equity-post-HCT-survival-predictions/train.csv')
    test_path = Path('../input/equity-post-HCT-survival-predictions/test.csv')
    subm_path = Path('../input/equity-post-HCT-survival-predictions/sample_submission.csv')

    batch_size = 32768
    early_stop = 50
    penalizer = 0.01
    n_splits = 5

    weights = [2, 1, 6, 3, 6, 3, 6, 6]

    ctb_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'reg_lambda': 8.0,
        'depth': 8
    }

    lgb_params = {
        'objective': 'regression',
        'min_child_samples': 32,
        'num_iterations': 6000,
        'learning_rate': 0.03,
        'extra_trees': True,
        'reg_lambda': 8.0,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'max_depth': 8,
        'device': 'cpu',
        'max_bin': 128,
        'verbose': -1,
        'seed': 42
    }

    xgb_params =  {
        'learning_rate': 0.06, 
        'max_depth': 7, 
        'n_estimators': 211, 
        'subsample': 0.80, 
        'colsample_bytree': 0.826, 
        'reg_alpha': 0.55, 
        'reg_lambda': 6.78,
        'seed': 42}

    cox1_params = {
        'grow_policy': 'Depthwise',
        'min_child_samples': 8,
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'reg_lambda': 8.0,
        'depth': 8
    }

    cox2_params = {
        'grow_policy': 'Lossguide',
        'min_child_samples': 2,
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'reg_lambda': 8.0,
        'num_leaves': 32,
        'depth': 8
    }


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
            "big_age",
            "year_hct",
            "is_cyto_score_same",
            "age_ts",
            "age_bin",
            'efs',
            'efs_time'
        ]

        for col in df.columns:

            if col in num_cols:
                df = df.with_columns(pl.col(col).fill_null(-1).cast(pl.Float32))  

            else:
                df = df.with_columns(pl.col(col).fill_null('Unknown').cast(pl.String))  

        return df.with_columns(pl.col('ID').cast(pl.Int32))
    #'is_cyto_score_same'と"year_hct"だけしか元のノートでは使われていなかった
    def engineer(self, df):
        # 年齢区分の境界値を定義（pd.cut の代替）
        bins = [0, 0.0441, 16, 30, 50, 100]
    
        df = df.with_columns([
            # sex_match_bool:
            # まず、sex_match を文字列にキャストし "-" で分割後、0番目と1番目が等しいか判定。
            # sex_match が null の場合は null を設定。
            pl.when(pl.col("sex_match").is_null())
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(
              pl.col("sex_match").cast(pl.Utf8)
              .str.split("-")
              .arr.get(0) == pl.col("sex_match").cast(pl.Utf8)
              .str.split("-")
              .arr.get(1)
            ).alias("sex_match_bool"),
        
            # big_age: age_at_hct > 16
            (pl.col("age_at_hct") > 16).alias("big_age"),
        
            # year_hct: 2019 の場合は 2020 に置き換え
            pl.when(pl.col("year_hct") == 2019)
              .then(2020)
              .otherwise(pl.col("year_hct"))
              .alias("year_hct"),
        
            # is_cyto_score_same: cyto_score と cyto_score_detail が等しければ 1、違えば 0
            (pl.col("cyto_score") == pl.col("cyto_score_detail"))
              .cast(pl.Int32)
              .alias("is_cyto_score_same"),
        
            # strange_age: age_at_hct が 0.044 と等しいかどうか
            (pl.col("age_at_hct") == 0.044).alias("strange_age"),
        
            pl.when(pl.col("age_at_hct") < bins[1])
              .then(pl.lit(f"[{bins[0]}, {bins[1]})"))
              .when((pl.col("age_at_hct") >= bins[1]) & (pl.col("age_at_hct") < bins[2]))
              .then(pl.lit(f"[{bins[1]}, {bins[2]})"))
              .when((pl.col("age_at_hct") >= bins[2]) & (pl.col("age_at_hct") < bins[3]))
              .then(pl.lit(f"[{bins[2]}, {bins[3]})"))
              .when((pl.col("age_at_hct") >= bins[3]) & (pl.col("age_at_hct") < bins[4]))
              .then(pl.lit(f"[{bins[3]}, {bins[4]})"))
              .when((pl.col("age_at_hct") >= bins[4]) & (pl.col("age_at_hct") <= bins[5]))
              .then(pl.lit(f"[{bins[4]}, {bins[5]})"))
              .otherwise(pl.lit(None).cast(pl.Utf8))
              .alias("age_bin"),
        
             # age_ts: age_at_hct / donor_age
            (pl.col("age_at_hct") / pl.col("donor_age")).alias("age_ts")
        ])
    
        #最後に、year_hct から 2000 を引く
        df = df.with_column(
            (pl.col("year_hct") - 2000).alias("year_hct")
        )
    
        return df

    def apply_fe(self, path):

        df = self._load_data(path)   
        df = self._update_hla_columns(df)   
        df = self.engineer(df)                  
        df = self._cast_datatypes(df)        
        df = df.to_pandas()
        
        
        cat_cols = [col for col in df.columns if df[col].dtype == pl.String]

        return df, cat_cols

fe = FE(CFG.batch_size)

train_data, cat_cols = fe.apply_fe(CFG.train_path)

class Targets:

    def __init__(self, data, cat_cols, penalizer, n_splits):
        
        self.data = data
        self.cat_cols = cat_cols
        
        self._length = len(self.data)
        self._penalizer = penalizer
        self._n_splits = n_splits

    def _prepare_cv(self):
        
        oof_preds = np.zeros(self._length)
            
        cv = KFold(n_splits=self._n_splits, shuffle=True, random_state=42)

        return cv, oof_preds

    def validate_model(self, preds, title):
            
        y_true = self.data[['ID', 'efs', 'efs_time', 'race_group']].copy()
        y_pred = self.data[['ID']].copy()
        
        y_pred['prediction'] = preds
            
        c_index_score = score(y_true.copy(), y_pred.copy(), 'ID')
        print(f'Overall Stratified C-Index Score for {title}: {c_index_score:.4f}')

    def create_target1(self):  

        '''
        Constant columns are dropped if they exist in a fold. Otherwise, the code produces error:

        delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation: 
        https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
        '''

        cv, oof_preds = self._prepare_cv()

        # Apply one hot encoding to categorical columns
        data = pd.get_dummies(self.data, columns=self.cat_cols, drop_first=True).drop('ID', axis=1) 

        for train_index, valid_index in cv.split(data):

            train_data = data.iloc[train_index]
            valid_data = data.iloc[valid_index]

            # Drop constant columns if they exist
            train_data = train_data.loc[:, train_data.nunique() > 1]
            valid_data = valid_data[train_data.columns]

            cph = CoxPHFitter(penalizer=self._penalizer)
            cph.fit(train_data, duration_col='efs_time', event_col='efs')
            
            oof_preds[valid_index] = cph.predict_partial_hazard(valid_data)              

        self.data['target1'] = oof_preds 
        self.validate_model(oof_preds, 'Cox') 

        return self.data

    def create_target2(self):        

        cv, oof_preds = self._prepare_cv()

        for train_index, valid_index in cv.split(self.data):

            train_data = self.data.iloc[train_index]
            valid_data = self.data.iloc[valid_index]

            kmf = KaplanMeierFitter()
            kmf.fit(durations=train_data['efs_time'], event_observed=train_data['efs'])
            
            oof_preds[valid_index] = kmf.survival_function_at_times(valid_data['efs_time']).values

        self.data['target2'] = oof_preds  
        self.validate_model(oof_preds, 'Kaplan-Meier')

        return self.data

    def create_target3(self):        

        cv, oof_preds = self._prepare_cv()

        for train_index, valid_index in cv.split(self.data):

            train_data = self.data.iloc[train_index]
            valid_data = self.data.iloc[valid_index]
            
            naf = NelsonAalenFitter()
            naf.fit(durations=train_data['efs_time'], event_observed=train_data['efs'])
            
            oof_preds[valid_index] = -naf.cumulative_hazard_at_times(valid_data['efs_time']).values

        self.data['target3'] = oof_preds  
        self.validate_model(oof_preds, 'Nelson-Aalen')

        return self.data

    def create_target4(self):

        self.data['target4'] = self.data.efs_time.copy()
        self.data.loc[self.data.efs == 0, 'target4'] *= -1

        return self.data

class MD:
    
    def __init__(self, data, cat_cols, early_stop, penalizer, n_splits):
        
        self.targets = Targets(data, cat_cols, penalizer, n_splits)
        
        self.data = data
        self.cat_cols = cat_cols
        self._early_stop = early_stop

    def create_targets(self):

        self.data = self.targets.create_target1()
        self.data = self.targets.create_target2()
        self.data = self.targets.create_target3()
        self.data = self.targets.create_target4()

        return self.data

md = MD(train_data, cat_cols, CFG.early_stop, CFG.penalizer, CFG.n_splits)
train_data = md.create_targets()

def validate(model_target, data, cv, weights, csv_name):
    for col in cat_cols:
        data[col] = data[col].astype('category')
    
    X = data.drop(['ID', 'efs', 'efs_time', 'target1', 'target2', 'target3', 'target4'], axis=1)
    y = data[['ID','efs', 'efs_time', 'target1', 'target2', 'target3', 'target4', 'race_group']].copy()

    fold_scores = {}
    target_scores = {}
    
    for fold, (train_index, valid_index) in enumerate(tqdm.tqdm(cv.split(X, y), total=cv.get_n_splits(), desc="Folds")):
        models = []
        oof_preds = np.zeros(len(valid_index), len(model_target))
        for i, (model, target) in enumerate(model_target):
            x_train, x_early, y_train, y_early = train_test_split(X.iloc[train_index], y[target].iloc[train_index], test_size=0.2, random_state=42,shuffle=True)

            if model.__class__.__name__ == 'CatBoostRegressor':
                model.fit(x_train, y_train, eval_set=(x_early, y_early),  early_stopping_rounds=CFG.early_stop, verbose=0)
            elif model.__class__.__name__ == 'XGBRegressor':
                model.fit(x_train, y_train, eval_set=[(x_early, y_early)],eval_metric='rmse',callbacks=[xgb.callback.EarlyStopping(rounds=CFG.early_stop)], verbose=0)
            elif model.__class__.__name__ == 'LGBMRegressor':
                model.fit(x_train, y_train, eval_set=[(x_early, y_early)], eval_metric='rmse',callbacks=[lgb.early_stopping(CFG.early_stop, verbose=0), lgb.log_evaluation(0)])
            else:
                model.fit(X.iloc[train_index], y[target].iloc[train_index])


            preds = data[['ID']].iloc[valid_index].copy()
            preds['prediction'] = model.predict(X.iloc[valid_index])
            oof_preds[:, i] = preds['prediction'].values

            s = score(y[['ID','efs', 'efs_time', 'race_group']].iloc[valid_index], preds, 'ID')
            # target毎のスコアを辞書に格納（キーにモデル名、ターゲット名、fold番号を含める）
            key = f"{model.__class__.__name__}_{target}_fold{fold}"
            target_scores[key] = s
            models.append(model)
            print("{fold}フォールド目の{model}の{target}のスコア: {s:.4f}".format(fold=fold, model=model.__class__.__name__,
                                                                target=target, s=s))
        
        # 複数モデルのアンサンブル
        #ensemble_model = VotingRegressor(estimators=[(f"{model.__class__.__name__}_{i}", model) for i, model in enumerate(models)], weights=weights)
        # アンサンブルの予測・スコア計算

        ens_preds = data[['ID']].iloc[valid_index].copy()
        ens_preds['prediction'] = oof_preds @ weights
        fold_score = score(y[['ID','efs', 'efs_time', 'race_group']].iloc[valid_index], ens_preds,'ID')
        fold_scores[f"fold_{fold}"] = fold_score

    fold_scores_df = pd.DataFrame(list(fold_scores.items()), columns=["fold", "score"])
    target_scores_df = pd.DataFrame(list(target_scores.items()), columns=["model_target_fold", "score"])
 
    # fold_scores_df にグループ名を追加して、キーの列名を統一する
    fold_scores_df["group"] = "fold_scores"
    fold_scores_df.rename(columns={"fold": "id"}, inplace=True)

    # target_scores_df にグループ名を追加して、キーの列名を統一する
    target_scores_df["group"] = "target_scores"
    target_scores_df.rename(columns={"model_target_fold": "id"}, inplace=True)

    # 2 つの DataFrame を縦方向に結合
    combined_df = pd.concat([fold_scores_df, target_scores_df], axis=0)

    # CSV ファイルとして出力
    combined_df.to_csv("{csv_name}.csv", index=False)
    print("CSVファイル{csv_name}の出力が完了しました。")

def get_dummy(data, columns=cat_cols):
  return pd.get_dummies(data, columns=columns)
random_params = {'max_depth': 35, 'n_estimators': 183, 'min_samples_split': 9, 'min_samples_leaf': 7, 'max_leaf_nodes': 15}


lgb_model = lgb.LGBMRegressor(**CFG.lgb_params)
cat_model = CatBoostRegressor(**CFG.ctb_params, verbose=0, cat_features=cat_cols)
xgb_model = xgb.XGBRegressor(**CFG.xgb_params,enable_categorical=True)
cox_model_1 = CatBoostRegressor(**CFG.cox1_params, verbose=0, cat_features=cat_cols)
cox_model_2 = CatBoostRegressor(**CFG.cox2_params, verbose=0, cat_features=cat_cols)
random_model = RandomForestRegressor(**random_params, random_state=42)
random_model = Pipeline(steps=[('get_dummy',FunctionTransformer(get_dummy)),('regressor',random_model)])


model_target = [[cat_model, "target1"], [lgb_model, "target1"], [cat_model, "target2"], [lgb_model, "target2"], [cat_model, "target3"], [lgb_model, "target3"],[cox_model_1, "target4"], [cox_model_2, "target4"]]
model_and_target =  [[random_model, 'efs_time'], [cat_model, "target2"], [lgb_model, "target2"], [cat_model, "target3"], [lgb_model, "target3"],[cox_model_1, "target4"], [cox_model_2, "target4"]]

weights = [2, 6, 3, 6, 3, 6, 6]

target = Targets(train_data, cat_cols, CFG.penalizer, CFG.n_splits)

cv, _ = target._prepare_cv()
validate(model_target, train_data, cv, CFG.weights, str([model_target, weights]))
    