import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

nvda = yf.Ticker('NVDA')

nvda = nvda.history(period="max")

del nvda["Dividends"]
del nvda["Stock Splits"]
nvda = nvda.loc['2000-01-01':].copy()

# nvda.plot.line(y="Close", use_index=True)
# plt.show()

nvda["Next_Day"] = nvda["Close"].shift(-1)
nvda["Target"] = (nvda["Next_Day"] > nvda["Close"]).astype(int)
# print(nvda.info)

model = RandomForestClassifier(n_estimators=200,min_samples_split=2,random_state=3)

train = nvda.iloc[:-100]
test = nvda.iloc[-100:]

predictors = ["Open","High","Low","Close","Volume"]
# model.fit(train[predictors],train["Target"])

# predictions = model.predict(test[predictors])
# predictions = pd.Series(predictions,index=test.index)

# print(precision_score(test["Target"],predictions))

# combined = pd.concat([test["Target"],predictions],axis=1)
# combined.plot()
# plt.show()

def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"])
    predictions = model.predict(test[predictors])
    predictions = pd.Series(predictions,index=test.index,name="Predictions")
    combined = pd.concat([test["Target"],predictions],axis=1)
    return combined

def backtest(data,model,predictors,start=2500,step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(nvda,model,predictors)
print(predictions["Predictions"].value_counts())
print(predictions["Predictions"].value_counts()/predictions.shape[0])

print(precision_score(predictions["Target"],predictions["Predictions"]))

horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = nvda.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    nvda[ratio_column] = nvda["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    nvda[trend_column] = nvda.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]
    
nvda = nvda.dropna()
