#@title 後期課題(python) 本データ
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # データセット分割用
np.set_printoptions(suppress=True)#指数表記禁止

import IPython
def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)

# 特徴量重要度を棒グラフでプロットする関数 
def plot_feature_importance(df): 
  n_features = len(df)                              # 特徴量数(説明変数の個数) 
  df_plot = df.sort_values('importance')            # df_importanceをプロット用に特徴量重要度を昇順ソート 
  f_importance_plot = df_plot['importance'].values  # 特徴量重要度の取得 
  plt.barh(range(n_features), f_importance_plot, align='center') 
  cols_plot = df_plot['feature'].values             # 特徴量の取得 
  plt.yticks(np.arange(n_features), cols_plot)      # x軸,y軸の値の設定
  plt.xlabel('Feature importance')                  # x軸のタイトル
  plt.ylabel('Feature')                             # y軸のタイトル


#データよみこみ
putibankdata = pd.read_csv(filepath_or_buffer="/content/bank-additional.csv", encoding="ms932", sep=";")
fullbankdata = pd.read_csv(filepath_or_buffer="/content/bank-additional-full.csv", encoding="ms932", sep=";")


#duration,pdaysの列を削除
#putidata = fullbankdata.drop(['duration','pdays','month','day_of_week'], axis=1)
putidata = fullbankdata.drop(['duration','pdays'], axis=1)

#変数を数値化
#2列目　job（仕事）
job_le = LabelEncoder()
putidata['job'] = job_le.fit_transform(putidata['job'])

#3列目　marital（結婚してるか）
marital_le = LabelEncoder()
putidata['marital'] = marital_le.fit_transform(putidata['marital'])

#4列目 education（教育水準）
edu_le = LabelEncoder()
putidata['education'] = edu_le.fit_transform(putidata['education'])

#5列目 default（破産したかどうか）
default_le = LabelEncoder()
putidata['default'] = default_le.fit_transform(putidata['default'])

#6列目 housing（家があるか）
house_le = LabelEncoder()
putidata['housing'] = house_le.fit_transform(putidata['housing'])

#7列目 loan（ローンがあるかどうか）
loan_le = LabelEncoder()
putidata['loan'] = loan_le.fit_transform(putidata['loan'])

#8列目 contact（客との接触手段）
contact_le = LabelEncoder()
putidata['contact'] = contact_le.fit_transform(putidata['contact'])

#8列目 poutcome（前回のキャンペーンが成功したかどうか）
poutcome_le = LabelEncoder()
putidata['poutcome'] = poutcome_le.fit_transform(putidata['poutcome'])

# month(持ち掛けた月)
month_le = LabelEncoder()
putidata['month'] = month_le.fit_transform(putidata['month'])

#day_of_week(話をした曜日)
day_le = LabelEncoder()
putidata['day_of_week'] = month_le.fit_transform(putidata['day_of_week'])


#最終列 y（預金契約をしたかどうか）
y_le = LabelEncoder()
putidata['y'] = y_le.fit_transform(putidata['poutcome'])

#print(putidata)
#いらなさそうな列を削除
#last_putidata = putidata.drop([])


# 説明変数,目的変数, テストケース分割
X = putidata.drop('y',axis=1).values # 説明変数
y = putidata['y'].values             # 目的変数
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=2)

# 学習に使用するデータを設定
putilgb_train = lgb.Dataset(X_train, y_train)
putilgb_eval = lgb.Dataset(X_test, y_test, reference=putilgb_train) 

# LightGBM parameters
params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        #'objective': 'regression', # 目的 : 回帰  
        #'objective':'mape',
        'metric': {'rmse'}, # 評価指標 : rsme(平均二乗誤差の平方根) 
}

# モデルの学習
model = lgb.train(params,
                  train_set=putilgb_train, # トレーニングデータの指定
                  valid_sets=putilgb_eval, # 検証データの指定
                  )




## 特徴量重要度の算出 (データフレームで取得)
cols = list(putidata.drop('y',axis=1).columns)       
# 特徴量重要度の算出方法 'gain'(推奨) : トレーニングデータの損失の減少量を評価
f_importance = np.array(model.feature_importance(importance_type='gain')) # 特徴量重要度の算出 //
f_importance = f_importance / np.sum(f_importance) # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
display(df_importance)
plot_feature_importance(df_importance)
