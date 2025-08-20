import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

cardataset=pd.read_csv('C:/Users/veere/OneDrive/Desktop/carprediction_streamlit/data/CAR_DETAILS_FROM_CAR_DEKHO_AUGMENTED.csv')
print(cardataset.head())
print(cardataset.info())
print(cardataset['fuel'].value_counts())
print(cardataset['seller_type'].value_counts())
print(cardataset['transmission'].value_counts())
print(cardataset['owner'].value_counts())


cardataset['fuel'] = cardataset['fuel'].map({'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4})
cardataset['seller_type'] = cardataset['seller_type'].map({'Individual':0,'Dealer':1,'Trustmark Dealer':2})
cardataset['transmission'] = cardataset['transmission'].map({'Manual':0,'Automatic':1})
cardataset['owner'] = cardataset['owner'].map({'First Owner':0,'Second Owner':1,'Third Owner':2,'Fourth & Above Owner':3,'Test Drive Car':4})


cardataset['brand'] = cardataset['name'].str.split().str[0]
le = LabelEncoder()
cardataset['brand'] = le.fit_transform(cardataset['brand'])

x= cardataset[['brand', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
y = cardataset['selling_price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
'''lr=LinearRegression()
lr.fit(x_train,y_train)

ytrain_pred=lr.predict(x_train)
r2_train=r2_score(y_train,ytrain_pred)
print("r2 train: ",r2_train)
plt.scatter(y_train,ytrain_pred,alpha=0.3)


ytest_pred=lr.predict(x_test)
r2_test=r2_score(y_test,ytest_pred)
print("r2 test: ",r2_test)
plt.scatter(y_test,ytest_pred,alpha=0.3)
print(cardataset['year'].max())

joblib.dump(lr,"model/model.pkl")
'''
'''from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)

print("RF Train R²:", rf.score(x_train, y_train))
print("RF Test R²:", rf.score(x_test, y_test))

joblib.dump({"model": rf, "label_encoder": le}, "model/model.pkl")'''

xgb = XGBRegressor(
    n_estimators=500,      # number of trees
    learning_rate=0.05,    # step size shrinkage
    max_depth=6,           # depth of each tree
    subsample=0.8,         # sample ratio
    colsample_bytree=0.8,  # feature sample ratio
    random_state=42,
    objective='reg:squarederror'
)


xgb.fit(x_train, y_train)
y_pred_train = xgb.predict(x_train)
y_pred_test = xgb.predict(x_test)

print("Train R²:", r2_score(y_train, y_pred_train))
print("Test R²:", r2_score(y_test, y_pred_test))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

joblib.dump({"model": xgb, "label_encoder": le}, "model/model.pkl")