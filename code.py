#%% 套件安裝
import numpy as np
import pandas as pd
#%%　檔案讀取
rate = pd.read_csv('rating.csv')
#%% 觀察資料遺失值
print('Missing value rate')
for col in rate.columns:
    miss_rate = rate[col].isnull().sum() / rate.shape[0]
    miss_bool = (miss_rate == 0 )
    print("{:>15} {:>7.2f} % {}".format(col, miss_rate*100, miss_bool))
#%%　評分分數資料處理
rate.rating.value_counts() #評分次數分布
rate = rate[rate.rating>0] #刪除評分為-1資料
#%% 被觀看動漫及用戶資料觀察
rate.anime_id.value_counts() #動漫被評分次數
rate.user_id.value_counts() #用戶評分次數
#%% 檢查稀疏矩陣
def sparsity(df):
    unique_user = df.user_id.value_counts()
    unique_anime = df.anime_id.value_counts()
num_user = len(unique_user)
    num_anime = len(unique_anime)
    sparse_rate = (1- (df.shape[0] / (num_user * num_anime))) * 100
    print("Unique users/anime: {}/{}".format(num_user, num_anime))
    print("Matrix sparse rate: {:.2f} %".format(sparse_rate))
    return unique_user, unique_anime

unique_user, unique_anime = sparsity(rate)
#%% 刪除不常見用戶及動漫資料
user_drop = unique_user[unique_user < 9].index # 13501 uncommon users
anime_drop = unique_anime[unique_anime < 5].index # 1901 uncommon animes

rate_df1 = rate.set_index('user_id', inplace=False)
rate_df2 = rate_df1.drop(user_drop, axis=0)

unique_anime2 = rate_df2.anime_id.value_counts()
anime_drop2 = unique_anime2[unique_anime2 < 5].index

rate_df2.reset_index(level='user_id', inplace=True)
rate_df2.set_index('anime_id', inplace=True)

rate_df3 = rate_df2.drop(anime_drop2, axis=0)
rate_df3.reset_index(level='anime_id', inplace=True)

_, _ = sparsity(rate_df3)
#%% 用戶資料抽樣
rate_df3.to_csv('all_data.csv') #將前述清理資料做輸出
all_data= pd.read_csv('all_data.csv')
alluserdata=rate_df3.user_id.value_counts()
alluserdata.to_csv('alluserdata.csv') #將用戶編號做輸出(過程用EXCEL保留用戶欄位)
sampleuser=alluserdata.sample(frac=0.75,axis=0) #抽出75%用戶
sampleuser.to_csv('sampleuser.csv') #讀出用戶，並新增欄位全部補1
sampleuser= pd.read_csv('sampleuser.csv')
#%% 合併抽樣用戶與清理過後Data
alldata =pd.merge(rate_df3,sampleuser,on="user_id",how="outer")
#%% 測試資料
tset_data = alldata[alldata.keep!=1]
tset_data=tset_data.drop("keep", axis = 1)
train_data.to_csv('train_data.csv')
#%% 訓練資料
train_data = alldata[alldata.keep==1]
train_data=train_data.drop("Unnamed: 0", axis = 1)
tset_data.to_csv('test_data.csv')
(二)KNN
#%% 套件安裝
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import surprise
from surprise import KNNBasic
from surprise import Dataset
from surprise.accuracy import rmse
from surprise.model_selection.split import train_test_split
from surprise import SVD, Dataset, Reader, NormalPredictor, KNNBaseline
#%% 資料讀取
sample_data = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
sample_data=sample_data.drop("Unnamed: 0", axis = 1)
test=test.drop("Unnamed: 0", axis = 1)
#%% Creat dataframe
#%% Shuffle and Split
sample_data = shuffle(sample_data)
thre = round(len(sample_data)*0.8)

train = sample_data[0: thre]
valid = sample_data[thre:]
#%% Creat dataframe
reader = Reader(rating_scale=(1, 10))  # claim the input format and rating scale
data_folds = Dataset.load_from_df(sample_data[['user_id', 'anime_id', 'rating']], reader)
train_folds = Dataset.load_from_df(train[['user_id', 'anime_id', 'rating']], reader)
valid_folds = Dataset.load_from_df(valid[['user_id', 'anime_id', 'rating']], reader)
test_folds = Dataset.load_from_df(test[['user_id', 'anime_id', 'rating']], reader)
#%% 交叉驗證
#%% Grid Search
param_grid = {'k': [5],
              'sim_options': {'name': ['pearson'],
                              'user_based': [False]
                             }
             }
grid = GridSearchCV(KNNBaseline,param_grid, measures=['RMSE'], cv=5, refit=False, joblib_verbose=3)
grid.fit(train_folds)
print(grid.best_score['rmse'])
print(grid.best_params['rmse'])
#%% 模型評估
model = grid.best_estimator['rmse']
model.fit(train_folds.build_full_trainset())
#%% Predict validset
validset = valid_folds.build_full_trainset()
predictions = model.test(validset.build_testset())
print(accuracy.rmse(predictions))
#%% Predict testset
testset = test_folds.build_full_trainset()
predictions = model.test(testset.build_testset())
print(accuracy.rmse(predictions))
#%% 預測結果
#%% Prediction on full dataset
dataset = data_folds.build_full_trainset()
predictions = model.test(dataset.build_testset())
def get_top_n(predictions, n=10):
 top_n = defaultdict(list)
 for uid, iid, true_r, est, _ in predictions:
  top_n[uid].append((iid, est))
 for uid, user_ratings in top_n.items():
  user_ratings.sort(key=lambda x: x[1], reverse=True)
     
  top_n[uid] = user_ratings[:n]

 return top_n

top_n = get_top_n(predictions, n=5)
top_n['30210']
top_n
(三)Funk SVD
#%% 套件安裝
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from surprise import SVD, Dataset, Reader, NormalPredictor, KNNBaseline
from surprise import accuracy
from surprise.model_selection import cross_validate, KFold, GridSearchCV
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#%% 資料讀取
sample_data = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
all_data= pd.read_csv('all_data.csv')
sample_data=sample_data.drop("Unnamed: 0", axis = 1)
test=test.drop("Unnamed: 0", axis = 1)
all_data=all_data.drop("Unnamed: 0", axis = 1)
#%% Shuffle and Split
sample_data = shuffle(sample_data)
thre = round(len(sample_data)*0.8)

train = sample_data[0: thre]
valid = sample_data[thre:]
#%% Creat dataframe
reader = Reader(rating_scale=(1, 10))  # claim the input format and rating scale

data_folds = Dataset.load_from_df(sample_data[['user_id', 'anime_id', 'rating']], reader)
train_folds = Dataset.load_from_df(train[['user_id', 'anime_id', 'rating']], reader)
valid_folds = Dataset.load_from_df(valid[['user_id', 'anime_id', 'rating']], reader)
test_folds = Dataset.load_from_df(test[['user_id', 'anime_id', 'rating']], reader)
#%% 交叉驗證
#%% Grid Search
param_grid = {'n_epochs': list(range(5, 100, 5)), 'lr_all': [0.01, 0.005],
              'reg_all': [0.4], 'n_factors': [100]}
grid = GridSearchCV(SVD, param_grid, measures=['RMSE'], cv=5, refit=False, joblib_verbose=3)
grid.fit(train_folds)
# Get best RMSE score
print(grid.best_score['rmse'])
print(grid.best_params['rmse'])
#%% 模型評估
#%% Train on full trainset by the best hyper-parameters
model = grid.best_estimator['rmse']
model.fit(data_folds.build_full_trainset())
#%% Get latent factors of user and item
model.pu # user factors
model.qi # item factors
#%% Predict validset
validset = valid_folds.build_full_trainset()
predictions = model.test(validset.build_testset())
print(accuracy.rmse(predictions))
#%% Predict testset
testset = test_folds.build_full_trainset()
predictions = model.test(testset.build_testset())
print(accuracy.rmse(predictions))
#%% 預測結果
#%% Prediction on full dataset
dataset = data_folds.build_full_trainset()
predictions = model.test(dataset.build_testset())
def get_top_n(predictions, n=10):
 top_n = defaultdict(list)
 for uid, iid, true_r, est, _ in predictions:
  top_n[uid].append((iid, est))
 for uid, user_ratings in top_n.items():
  user_ratings.sort(key=lambda x: x[1], reverse=True)
     
  top_n[uid] = user_ratings[:n]

 return top_n

top_n = get_top_n(predictions, n=5)
top_n['30210']
#%% (四)FM
#%% 套件安裝
import numpy as np
import pandas as pd
import xlearn as xl
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
#%% 資料讀取
sample_data = pd.read_csv('train_data.csv')
sample_data=sample_data.drop("Unnamed: 0", axis = 1)
test_data = pd.read_csv('test_data.csv')
test_data=test_data.drop("Unnamed: 0", axis = 1)
sample_data.user_id= sample_data.user_id.astype(str)
sample_data.anime_id= sample_data.anime_id.astype(str)
#%% 新增虛擬變數
raw_df_cat = sample_data[['user_id', 'anime_id']]
raw_df_dummy = pd.get_dummies(raw_df_cat,prefix=['user', 'anime'], dummy_na=True,sparse=True)
cleaned = pd.concat([sample_data[['rating']], raw_df_dummy], axis=1, copy=False)
del raw_df_cat, raw_df_dummy
boundary = round(len(cleaned) * 0.8)
train = cleaned.loc[:boundary]
test = cleaned.loc[boundary:]
#%% 模型訓練
fm_model = xl.create_fm()
fm_model.setTrain("train_rating.csv")
fm_model.setValidate("test_rating.csv")
#%% 交叉驗證
param = {'task':'reg',
  'metric': 'rmse',
  'lr':0.1,
  'lambda':0.01,
  'epoch':1000,
  'opt':'adagrad',
  'k':100,
  'stop_window':3,
  'nthread':4,
  'block_size':512,
  }
fm_model.cv(param)
fm_model.fit(param, "./model/fm01.out")
#%% 預測結果
fm_model = xl.create_fm()
fm_model.setTest("./data/test_data.csv")
fm_model.predict("./model/fm01.out", ./output/output.txt")
#%% (五)動漫型態評分人數圖
import matplotlib.pyplot as plt 
df1=anime[anime['type']=='TV']
df2=anime[anime['type']=='Movie']
df3=anime[anime['type']=='OVA']
df4=anime[anime['type']=='Speacial']
df5=anime[anime['type']=='ONA']
df6=anime[anime['type']=='Music']
plt.hist([df1.rating,df2.rating,df3.rating,df4.rating,df5.rating,df6.rating],
         label=['TV','Movie','OVA','Speacial','ONA','Music'],stacked=True)
plt.legend()
plt.xlabel('rating')
plt.ylabel('number of member')
plt.show()
#%% (六)評分平均分布圖
import matplotlib.pyplot as plt 
%matplotlib inline 
ave_rate['rating_x'].hist(bins=50)
#%% (七)評分次數分布圖
ave_rate['number_of_ratings'].hist(bins=60)
#%% (八)評分平均分數與被評分次數散佈圖
import seaborn as sns 
sns.jointplot(x='rating_x', y='number_of_ratings', data=ave_rate)
#%% (九)rating用戶評分分布圖
import pandas as pd
rate = pd.read_csv('D:/rating.csv')
import matplotlib.pyplot as plt
import matplotlib.font_manager
a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
 for i in a:
    print(i)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.bar(rate.rating.unique(),
        rate.rating.value_counts(), 
        width=0.5, 
        bottom=None, 
        align='center', 
        color=['cadetblue', 
               'darkturquoise', 
               'darkcyan', 
               'lightseagreen', 
               'mediumturquoise', 
               'turquoise'
               ])
plt.xticks(range(-1,11))
plt.xlabel('會員評分')
plt.ylabel('總評分次數')
plt.title('rating用戶評分分布圖')
plt.show()
#%% (十)train_data用戶評分分布圖
import pandas as pd
rate = pd.read_csv('D:/sampledata.csv')
import matplotlib.pyplot as plt
import matplotlib.font_manager
a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
 for i in a:
    print(i)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.bar(rate.rating.unique(),
        rate.rating.value_counts(), 
        width=0.5, 
        bottom=None, 
        align='center', 
        color=['cadetblue', 
               'darkturquoise', 
               'darkcyan', 
               'lightseagreen', 
               'mediumturquoise', 
               'turquoise'
               ])
plt.xticks(range(-1,11))
plt.xlabel('會員評分')
plt.ylabel('總評分次數')
plt.title('train_data用戶評分分布圖')
plt.show()
#%% (十一)test_rating用戶評分分布圖:
import pandas as pd
rate = pd.read_csv('D:/test_rating.csv')
import matplotlib.pyplot as plt
import matplotlib.font_manager
a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
 for i in a:
    print(i)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.bar(rate.rating.unique(),
        rate.rating.value_counts(), 
        width=0.5, 
        bottom=None, 
        align='center', 
        color=['cadetblue', 
               'darkturquoise', 
               'darkcyan', 
               'lightseagreen', 
               'mediumturquoise', 
               'turquoise'
               ])
plt.xticks(range(-1,11))
plt.xlabel('會員評分')
plt.ylabel('總評分次數')
plt.title('test_rating用戶評分分布圖')
plt.show()



