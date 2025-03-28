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


class OANDAClient:
    def __init__(self, access_token, account_id, environment="practice"):
        """
        Initialize the OANDA API client.
        :param access_token: (str) Your OANDA API token.
        :param account_id: (str) Your OANDA account ID.
        :param environment: (str) "practice" for demo, "live" for real trading.
        """
        self.account_id = account_id
        self.client = oandapyV20.API(access_token=access_token, environment=environment)
        self.db = DataDB_tracker()
        self.now = datetime.datetime.utcnow()
        self.check_if_close_trade_time()
        self.limit_order_checker()


    
    def count_trading_days(self, start_date, end_date):
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  
        return len(date_range)-1 

    def db_data_spitter(self, response):
        try:
            impor = response['orderFillTransaction']
            data =  {
                'pair':impor['instrument'],
                'execution_time': impor['time'],
                'type':impor['reason'],
                'direction': 1 if float(impor['units']) > 0 else -1,
                'units': impor['units'],
                'price':float(impor['tradeOpened']['price'])
                
            }
        except KeyError:
            impor = response['orderCreateTransaction']
            data =  {
                'pair':impor['instrument'],
                'execution_time': impor['gtdTime'],
                'type':impor['type'],
                'direction': 1 if float(impor['units']) > 0 else -1,
                'units': impor['units'],
                'price':float(impor['price'])
                
            }         
        return data

    def get_last_friday_execution_time():
        """Returns the last Friday at 21:01 UTC in ISO format."""
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        last_friday = now - timedelta(days=(now.weekday() + 3) % 7) 
        execution_time = last_friday.replace(hour=21, minute=1, second=0, microsecond=0)
        return execution_time.strftime("%Y-%m-%dT%H:%M:%S.000000Z")



        
    def limit_order_checker(self):
        """Checks all LIMIT orders in the database, updates them if filled, or deletes them if expired."""
        trades_list = self.db.query_all(DataDB_tracker.FOREX_COLL) 
    
        for trade in trades_list:
            if trade.get('type') != "LIMIT_ORDER":
                continue  
            
            instrument = trade.get('pair')
            units = trade.get('units')
            execution_time_str = trade.get('execution_time')
    
            if not instrument or not units or not execution_time_str:
                print(f"Skipping trade due to missing data: {trade}")
                continue
    

            try:
                execution_time_str = trade.get('execution_time')

                execution_time_str = execution_time_str.split('Z')[0]  
                execution_time_str = execution_time_str[:26]  
                execution_time_str += "Z"  
                
                execution_time = datetime.datetime.strptime(execution_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                execution_time = execution_time.replace(tzinfo=None)
            except ValueError:
                print(f"Invalid execution_time format for trade: {trade}")
                continue
    

            try:

                r = orders.OrderList(self.account_id)
                response = self.client.request(r)
                open_orders = response.get('orders', [])
        

                trade_still_open = any(
                    o.get('instrument') == instrument and str(o.get('units')) == str(units) and o.get('type') == "LIMIT"
                    for o in open_orders
                )
        

                current_time = self.now
                
                if trade_still_open and execution_time <= current_time:

                    self.db.delete_one(DataDB_tracker.FOREX_COLL, pair= instrument, units= str(units))
                    print(f"Deleted expired unfilled trade for {instrument}.")
        
                elif not trade_still_open:
                    trade = self.db.query_single(DataDB_tracker.FOREX_COLL, pair = instrument,units =  str(units))
                    if trade:
                        trade['type'] = "MARKET"
                        trade['execution_time'] = self.get_last_friday_execution_time()
        
                        self.db.delete_one(DataDB_tracker.FOREX_COLL, pair=instrument, units=str(units))

                        # Insert the modified trade
                        self.db.add_one(DataDB_tracker.FOREX_COLL, trade)

                        
                        print(f"Updated filled trade for {instrument} to MARKET.")

    
            except Exception as e:
                print(f"Error checking trade for {instrument}: {e}")
    
    def check_if_close_trade_time(self):
        
        trades = self.db.query_all(DataDB_tracker.FOREX_COLL)
        trades_that_need_2_be_closed = []

        now = self.now.date()
        
        for trade in trades:
            execution_time = trade["execution_time"].rstrip("Z")[:26] 
            execution_time = datetime.datetime.fromisoformat(execution_time)
            trading_days_elapsed = self.count_trading_days(execution_time, now)

            if trading_days_elapsed >= 5:
                trades_that_need_2_be_closed.append(trade)

        for trade in trades_that_need_2_be_closed:
            self.close_a_trade(trade['pair'])
            self.db.delete_one(DataDB_tracker.FOREX_COLL, pair = trade['pair'])
        
    def is_market_closed(self):
        """Check if current time is between Friday 4:30 PM and Monday 1 AM UTC."""
        if (self.now.weekday() == 4 and self.now.hour >= 16 and self.now.minute >= 30) or (self.now.weekday() == 5 or self.now.weekday() == 6) or (self.now.weekday() == 0 and self.now.hour < 1):
            return True
        return False


        
    def get_account_equity(self):
        """
        Fetch the account equity (balance including unrealized P/L).
        :return: (float) Account equity.
        """
        try:
            r = accounts.AccountSummary(self.account_id)
            response = self.client.request(r)
            return float(response["account"]["NAV"]) 
        except V20Error as e:
            print(f"Error fetching account equity: {e}")
            return None

            
            
    def get_current_price(self, currency_pair):
        """Get current price for the given currency pair"""
        try:
            params = {
                'instruments': currency_pair
            }
            request = pricing.PricingInfo(self.account_id, params=params)
            response = self.client.request(request)
            price = float(response['prices'][0]['closeoutBid'])
            return price
        except V20Error as e:
            print(f"Error fetching current price for {currency_pair}: {e}")
            return None

    

    def calculate_position_size(self, currency_pair, risk_percentage=1):
        """Calculate position size based on account equity and stop loss."""
        stop_loss_pips = 0.0200 if not currency_pair.endswith(("JPY", "TRY")) else 2.00
        pip_value = 1 if not currency_pair.endswith(("JPY", "TRY")) else 10

        balance = self.get_account_equity()
        if balance is None:
            return None

        risk_amount = balance * (risk_percentage / 100)
        current_price = self.get_current_price(currency_pair)
        if current_price is None:
            return None

        pip_value_adjusted = pip_value * current_price
        position_size = risk_amount / (stop_loss_pips * pip_value_adjusted)

        return round(position_size, 0) 


    def place_order(self, instrument, direction, risk_percentage=1):
        """
        Places an order. If the market is closed, a limit order is placed with an expiry before Monday 1:15 AM UTC.
        direction: 1 for long, -1 for short.
        """
        

        position = self.calculate_position_size(currency_pair=instrument, risk_percentage=risk_percentage)
        if position is None:
            print("Failed to calculate position size.")
            return None

        current_price = self.get_current_price(instrument)
        if current_price is None:
            print("Failed to fetch current price.")
            return None
    

        position *= -1 if direction == -1 else 1  
    

        if self.is_market_closed():
            order_type = "LIMIT"
            price = str(current_price)  
    

            now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
            next_monday = now + timedelta(days=(7 - now.weekday()))  
            expiry_time = next_monday.replace(hour=5, minute=15, second=0, microsecond=0)
    
            time_in_force = "GTD"
            gtd_time = expiry_time.strftime("%Y-%m-%dT%H:%M:%S.000000Z")  
        else:
            order_type = "MARKET"
            price = None
            time_in_force = "FOK"  
    
        # Construct order data
        order_data = {
            "order": {
                "instrument": instrument,
                "units": str(position),
                "type": order_type,
                "timeInForce": time_in_force,
                "positionFill": "DEFAULT"
            }
        }
    
        if price:
            order_data["order"]["price"] = price  
        if order_type == "LIMIT":
            order_data["order"]["gtdTime"] = gtd_time  
    

        try:
            r = orders.OrderCreate(self.account_id, data=order_data)
            response = self.client.request(r)
            print(response)
            print(f"Order placed: {instrument}")
    
            # Save order to database
            data = self.db_data_spitter(response)
            self.db.add_one(DataDB_tracker.FOREX_COLL, data)
    
            return response
        except V20Error as e:
            print(f"Error placing order: {e}")
            return None



    
    def close_a_trade(self, currency_pair):
        """Close the first open trade for the specified currency pair."""
        try:
            r = trades.OpenTrades(self.account_id)
            response = self.client.request(r)
            open_trades = response.get("trades", [])

            for trade in open_trades:
                if trade["instrument"] == currency_pair:
                    trade_id = trade["id"]
                    close_request = trades.TradeClose(self.account_id, tradeID=trade_id)
                    close_response = self.client.request(close_request)
                    print(f"trade closed")


                    return close_response
            print("No open trades found for", currency_pair)
            return None
        except V20Error as e:
            print(f"Error closing trade: {e}")
            return None
