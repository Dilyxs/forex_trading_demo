from features_dict import dict_for_features
from elements import for_prediction_2, get_market_sentiment
import pandas as pd
import numpy as np

from just_class import MultiModelClassifierPredict, OANDAClientFriday_Edition
from constants import  API_KEY, ACCOUNT_ID
import logging
logging.basicConfig(filename="Oanda_logging.log",
                            level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')




pairs = ["EUR_USD", "AUD_USD", "USD_JPY", "USD_MXN", "USD_ZAR", "USD_CHF", "USD_CAD", "GBP_USD", "NZD_USD"]



def get_confidence_values(pair):
    df = pd.read_csv("get_confidence.csv")
    c_xgb = float(df[df.pair == pair].c_xgb.iloc[0]) 
    c_lgb = float(df[df.pair == pair].c_lgb.iloc[0])  
    return c_xgb, c_lgb 




def trade_returner(df):

    df[['c_xgb', 'c_lgb']] = df['pair'].apply(lambda pair: pd.Series(get_confidence_values(pair)))
    
    #these 2 are like special conditions
    df['c_met'] = (df['Confidence_xgb'] > df['c_xgb']) & (df['Confidence_lgb'] > df['c_lgb'])
    df['cot_met'] = (df['Prediction_xgb'] == df['EMA_encoded_cot']) 


    df['same_side'] = (df.Prediction_xgb == df.Prediction_lgb) & (df.Prediction_xgb != 1)
    

    df['trade'] = np.select(
    [
        (df['same_side'] == True) & (df['Prediction_xgb'] == 2) &  (df.Prediction_xgb == df.market_direction) ,
        (df['same_side'] == True) & (df['Prediction_xgb'] == 0) &  (df.Prediction_xgb == df.market_direction) 
    ],
    [1, -1],
    default=0 
)



    df['super_trade'] = np.select(
    [
        (df['trade'] == 1) & df['c_met'] & df['cot_met'],
        (df['trade'] == -1) & df['c_met'] & df['cot_met']
    ],
    [1, -1],
    default=0
)
    return df

def df_returner(pairs):
    target = "future_close_encoded"
    dfs = []
    for pair in pairs:
        print(f"currently doing {pair}")
        features = dict_for_features[pair]
        df = for_prediction_2(pair)
        prediction_clas = MultiModelClassifierPredict(df, features, target, pair)
        df_final = prediction_clas.run_all_models()
        df_final = df_final[[x for x in df_final.columns if any(y in x for y in ["close", "Prediction", "Confidence", 'time'])]]
        df_final = df_final.drop(columns = ['News_CB Consumer Confidence', 'high_close', 'low_close', 'close_above_EMA'])
        df_mk = get_market_sentiment(pair)
        df_mk['Date'] = pd.to_datetime(df_mk['Date']).dt.normalize()
        df_final['time'] = pd.to_datetime(df_final['time']).dt.normalize()
        df_merged = pd.merge(df_final.iloc[-1:], df_mk[['Date', 'market_direction', 'EMA_encoded_cot']], 
                             left_on='time', right_on='Date', how='left')
        df_merged.drop(columns=['Date', 'future_close'], inplace=True)
        df_merged['pair'] = pair
    
        dfs.append(df_merged)
        
    df_merged = pd.concat(dfs)
    df_merged['market_direction'] = np.where(df_merged.market_direction == 1, 2, 0)
    df_merged['EMA_encoded_cot'] = np.where(df_merged.EMA_encoded_cot == 1, 2, 0)
    return df_merged





def for_trade_execution(df, trader):
    clean_df = df[df.trade != 0]
    
    for _, row in clean_df.iterrows():
        pair = row['pair']  
        direction = row['trade']
        super_direction = row['super_trade']


        if super_direction != 0:
            if super_direction <0:
                trader.open_trade(pair, direction = -1,risk_percentage = 2)

            elif super_direction >0:
                trader.open_trade(pair, direction = 1,risk_percentage = 2)

        else:
            trader.open_trade(pair, direction)


df = df_returner(pairs)
df = trade_returner(df)
df.to_csv("test_err.csv")
    
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

try:
    trader = OANDAClientFriday_Edition(API_KEY, ACCOUNT_ID) #always intialize, will always check if trade needs to be closed

    if trader.is_friday:
        logging.info('today is friday')
    else:
        for_trade_execution(df, trader) 

except Exception as e:
    print(f"Error executing trades - {e}")

