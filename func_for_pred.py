from features_dict import dict_for_features
from elements import df_for_news

from just_class import MultiModelClassifier




pairs = ["EUR_USD", "USD_CHF", 'USD_CAD', "GBP_USD", "AUD_USD", "USD_JPY", "NZD_USD" ,  
    "USD_MXN",
    "USD_ZAR"]
timeframe = "D"
target = "future_close_encoded"
for pair in pairs:
    print(f"currently doing {pair}")
    features = dict_for_features[pair]
    df = df_for_news(pair, timeframe)
    classifier = MultiModelClassifier(df, features, target, pair)
    df_a = classifier.run_all_models()
