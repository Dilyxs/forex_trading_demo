import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from functools import reduce
from mgdb_trade_tracker import DataDB_tracker
import os
import joblib   
import datetime
from datetime import timedelta
import datetime
import oandapyV20
import pytz
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.pricing as pricing
import logging


def db_data_spitter(response):
    impor = response['orderFillTransaction']
    data =  {
        'pair':impor['instrument'],
        'execution_time': impor['time'],
        'type':impor['reason'],
        'direction': 1 if float(impor['units']) > 0 else -1,
        'units': impor['units'],
        'price':float(impor['tradeOpened']['price'])
        
    }
    return data


class MultiModelClassifier:
    def __init__(self, df, features, target, pair, batch_size=16, epochs=130, confidence_threshold=0.9, timeframe="D", timesteps=10):
        self.df = df
        self.features = [feature for feature in features if feature in df.columns]
        self.target = target
        self.pair = pair
        self.batch_size = batch_size
        self.epochs = epochs
        self.confidence_threshold = confidence_threshold
        self.timeframe = timeframe
        self.timesteps = timesteps
        self.models = {}
        self.scaler = None
        self.forecast_steps = None
     #   self.freq = "D"
      #  self.prediction_length = 5  
       # self.start_train = pd.to_datetime(self.df.iloc[:1, :]['time'].values)[0]
       # self.start_test =   pd.to_datetime(df.iloc[round(0.75 * len(self.df)):, :]['time'].values)[0]
        self.model_dir = "xg_model"
    def _prepare_data(self):
        df = self.df.copy()
        X = df[self.features]
        y = df[self.target]

        rows_c = round(0.75 * len(df))
        X_train, X_test = X[:rows_c], X[rows_c:]
        y_train, y_test = y[:rows_c], y[rows_c:]
        df_test = df[rows_c:].copy()

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Save scaler
        scaler_path = os.path.join(self.model_dir, f"{self.pair}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved at: {scaler_path}")

        self.scaler = scaler

        return X_train_sc, X_test_sc, y_train, y_test, df_test
    
  
    def train_xgb_boost(self):
        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
        df_merged = pd.read_csv("optuma_given_param.csv")
        params = (df_merged[(df_merged.pair == self.pair) & (df_merged.model == "XBGboost")]['param']).values[0]
        params = ast.literal_eval(params)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)
        model.fit(X_train_sc, y_train)

        y_pred_prob_train = model.predict_proba(X_test_sc)
        y_pred_train = np.argmax(y_pred_prob_train, axis=1)

        df_test['Prediction_xgb'] = y_pred_train
        df_test['Confidence_xgb'] = np.max(y_pred_prob_train, axis=1)

        self.models['xgb'] = model

        accuracy = accuracy_score(y_test, y_pred_train)
        print(f"XGBoost Model accuracy: {accuracy * 100:.2f}%")

        # Save model
        model_path = os.path.join(self.model_dir, f"{self.pair}_xgb.json")
        model.save_model(model_path)
        print(f"XGBoost model saved at: {model_path}")

        return df_test

    def train_lgb_boost(self):
        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
        df_merged = pd.read_csv("optuma_given_param.csv")
        params = (df_merged[(df_merged.pair == self.pair) & (df_merged.model == "LBGboost")]['param']).values[0]
        params = ast.literal_eval(params)

        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_sc, y_train)

        y_pred_prob_train = model.predict_proba(X_test_sc)
        y_pred_train = np.argmax(y_pred_prob_train, axis=1)

        df_test['Prediction_lgb'] = y_pred_train
        df_test['Confidence_lgb'] = np.max(y_pred_prob_train, axis=1)

        self.models['lgb'] = model

        accuracy = accuracy_score(y_test, y_pred_train)
        print(f"LGB Model accuracy: {accuracy * 100:.2f}%")

        # Save model
        model_path = os.path.join(self.model_dir, f"{self.pair}_lgb.txt")
        model.booster_.save_model(model_path)
        print(f"LGB model saved at: {model_path}")

        return df_test

 
    def run_all_models(self):
        df_test_xgb = self.train_xgb_boost()
        df_test_lgb = self.train_lgb_boost()

        data_frames = [df_test_xgb, df_test_lgb]
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=self.df.index.name or self.df.index.names[0], how='outer'), data_frames)
        return df_merged
    
        






class MultiModelClassifierPredict:
    def __init__(self,df, features, target, pair, confidence_threshold=0.9):
        self.df = df
        self.features = [feature for feature in features if feature in df.columns]
        self.target = target
        self.pair = pair
        self.confidence_threshold = confidence_threshold
        self.scaler = None
        self.batch_size = 16
        self.timesteps = 10
        self.xgb_model = None
        self.lgb_model = None
        self.load_models()
        #self.forecast_steps = trained_model.forecast_steps

    def _prepare_data_for_prediction(self):
        X = self.df[self.features]
        X_sc = self.scaler.transform(X)
        return X_sc

    def load_models(self):
        with open(f"xg_model/{self.pair}_scaler.pkl", "rb") as f:
            self.scaler = joblib.load(f)

        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(f"xg_model/{self.pair}_xgb.json")

        self.lgb_model = lgb.Booster(model_file=f"xg_model/{self.pair}_lgb.txt")



    def predict_xgb_boost(self):
        X_sc = self._prepare_data_for_prediction()
        model = self.xgb_model

        y_pred_prob = model.predict_proba(X_sc)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        df_test = self.df.copy()
        df_test['Prediction_xgb'] = y_pred
        df_test['Confidence_xgb'] = np.max(y_pred_prob, axis=1)
        
        return df_test

    def predict_lgb_boost(self):
        X_sc = self._prepare_data_for_prediction()
        model = self.lgb_model
        
        y_pred_prob = model.predict(X_sc)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        df_test = self.df.copy()
        df_test['Prediction_lgb'] = y_pred
        df_test['Confidence_lgb'] = np.max(y_pred_prob, axis=1)
        
        return df_test

    def run_all_models(self):
        models = [ self.predict_xgb_boost(), self.predict_lgb_boost()
        ]
        
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=self.df.index.name or self.df.index.names[0], how='outer'), models)
        return df_merged

