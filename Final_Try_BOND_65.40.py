
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn import cross_validation


# In[2]:

# Function to convert time into timestamp 
def convert_epoch(series):
    #final_t = series.apply(lambda x: x.replace(x.split(" ")[0], '').strip())
    final_datetime = series.apply(lambda x: datetime.strptime(x, '%d%b%Y'))
    final_datetime1 = final_datetime.apply(lambda x: x.timestamp())
    return final_datetime1


# In[3]:

# Function for time in meta data
def convert_epoch_meta(series):
    #final_datetime = series.apply(lambda x: datetime.strptime(x, '%d%b%Y'))
    final_datetime1 = series.apply(lambda x: x.timestamp())
    return final_datetime1


# In[4]:

# Function to correct formats of date in meta data
def try_parsing_date(text):
    for fmt in ('%d%b%Y', '%d-%B%-y', '%d-%b-%y', '%d%B%Y'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')


# In[5]:

# Data Cleaning of part1 of train data 
def data_clean(data):
    le=LabelEncoder()
    data1 = data
    data1["side"] = le.fit_transform(data1["side"])
    data1["timeofday"] = le.fit_transform(data1["timeofday"])
    data1["weekday"] = data1["date"].apply(lambda x: ep_to_day(x))
    data1["weekday"] = le.fit_transform(data1["weekday"])
    return data1


# In[6]:

# Data Cleaning and preprocessing of meta data PART1
def meta_data_clean1(data):
    data1 = data
    head_id = ["ratingAgency1EffectiveDate", "ratingAgency2EffectiveDate", "maturity"]
    data1 = data1.drop(head_id, axis = 1)
    return data1


# In[7]:

# Data Cleaning and Preprocessing of meta data PART2
def meta_data_clean2(data):
    data1 = data.copy()
    le=LabelEncoder()
    new_head = list(data1.columns.values)
    new_head.remove("amtIssued")
    new_head.remove("amtOutstanding")
    new_head.remove("coupon")
    new_head.remove("isin")
    new_head.remove("ratingAgency1Rating")
    new_head.remove("ratingAgency2Rating")
    new_head.remove("paymentRank")
    new_head.remove("issue date")
    new_head.remove("couponFrequency")
    data1["issue date"] = data1["issue date"].fillna("10March2016")
    #data1["issue date"] = data1["issue date"].apply(lambda x : (x.split("-")[0]+x.split("-")[1]+"20"+x.split("-")[2])
    #                                                if '-' in x else x)
    #data1["issue date"] = try_parsing_date(data1["issue date"]).values
    l = []
    for x in data1["issue date"]:
        #print(x)
        l.append(try_parsing_date(x))
    data1 = data1.drop("issue date", axis = 1)
    data1["issue date"] = l
    data1["issue date"] = convert_epoch_meta(data1["issue date"]).values
    
    for x in new_head:
        data1[x] = le.fit_transform(data1[x])
    rate_head = ["ratingAgency1Rating", "ratingAgency2Rating"]
    for x in rate_head:
        data1[x] = data1[x].apply(lambda x: int(x.replace("rating", '')))
    data1["paymentRank"] = data1["paymentRank"].apply(lambda x: int(x.replace("paymentRank", '')))
    return data1
    


# In[8]:

# Function to convert timestamp to week day
import time
def ep_to_day(ep):
    return time.strftime('%a', time.localtime(ep))


# In[9]:

# XGBoost Model
def run_xgboost(X, Y, Xt):    
    dTrain = xgb.DMatrix(X,label=Y,missing=np.NAN)
    dtest=xgb.DMatrix(Xt)
    XT,XV,YT,YV = cross_validation.train_test_split(X,Y,test_size=0.15,random_state=50)
    dtrain = xgb.DMatrix(XT,label=YT, missing=np.NAN)
    dval = xgb.DMatrix(XV,label=YV, missing=np.NAN)
    #param = {'eta': 0.05, 'max_depth': 8,'eval_metric': 'auc', 'objective': 'binary:logistic', 'silent': 1}

    param={
        'objective':"reg:linear",
        'booster':"gbtree",
        'eval_metric':"rmse",
        'eta':0.1,
        'max_depth':6,
        #'colsample_bytree':0.7,
        #'min_child_weight':1,
        #'num_boost_round':20,
        'silent': 1
        }

    evallist = [(dtrain,'dtrain'),(dval, 'eval')]
    num_round = 200
    bst = xgb.train(param, dTrain, num_round, evals=evallist)
    ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    ypred = pd.DataFrame(ypred)
    return ypred


# ## --------------------------------------------

# ## Starting with the dataset

# In[68]:

data = pd.read_csv("dataset.csv")
meta_data = pd.read_csv("ML_Bond_metadata.csv")


# In[69]:

data.head()


# ## **
# 

# First, we have clean the given dataset
# Second, we have converted the given dataset to include the timeofday when on given dates trade are not done.
# This we have generated and stored in train_data_gen.csv. 
# So, for this code, we already had the generated file and so we are just mentioning the function by which we generated.
# ##### Making this train_data_gen (i.e., making data for the timeofday on given dates when no trade is done)

# In[10]:

def generate_train_data_gen(data):
    m_new=data.groupby(["isin","date","side","timeofday"]).volume.sum().reset_index()
    m_backup = m_new.copy()
    m_new["date"] = convert_epoch(m_new["date"]).values
    data_cl = data_clean(m_new)
    print(data_cl.head())
    tiger = list(data_cl.groupby(["isin","date"]))
    poss_combo = [(0,1), (0,0), (1,0), (1,1)]
    len_tiger = len(tiger)

    main_isin_list = []
    print("start")
    for i in range(len_tiger):
        if i%10000 == 0:
            print(i)
        side_l = list(tiger[i][1]["side"])
        timeofday_l = list(tiger[i][1]["timeofday"])
        combo_list = list(zip(side_l, timeofday_l))
        new_side_l = []
        new_timeofday_l = []
        for combo in poss_combo:
            if combo not in combo_list:
                new_side_l.append(combo[0])
                new_timeofday_l.append(combo[1])

        len_this = len(new_side_l)
        list_df = []
        weekday_l = tiger[i][1]["weekday"].iloc[0]
        if len_this!=0:
            list_df.append(tiger[i][1])
            tmp_df = pd.DataFrame(index = np.arange(len_this))    
            tmp_df["isin"] = tiger[i][0][0]
            tmp_df["date"] = tiger[i][0][1]
            tmp_df["side"] = new_side_l
            tmp_df["timeofday"] = new_timeofday_l
            tmp_df["volume"] = 0
            #print(tiger[i][1]["weekday"])
            tmp_df["weekday"] = weekday_l
            list_df.append(tmp_df)
            main_kl = pd.concat(list_df)
            main_isin_list.append(main_kl)
            #print(main_kl)
        else:
            main_isin_list.append(tiger[i][1])


    main_isin_df = pd.concat(main_isin_list)
    print(main_isin_df.shape)
    print(main_isin_df.head())
    main_isin_df.to_csv("train_data_gen.csv")
    
# Generating train_data_gen.csv
# Here we have already generated files in our previous codes, so just putting the function here and the way to generate it
generate_train_data_gen(data)


# In[11]:

main_isin_df = pd.read_csv("train_data_gen.csv")
print(main_isin_df.shape)
main_isin_df.head()


# In[71]:

main_isin_df = pd.read_csv("train_data_gen.csv")
print(main_isin_df.shape)
main_isin_df.head()

main_isin_df = main_isin_df.drop("Unnamed: 0", axis = 1)


# In[72]:

main_isin_df_group = list(main_isin_df.groupby("isin"))


# In[73]:

#try_12 = main_isin_df_group[0][1].copy()
#try_12.sort_values(by=['date', 'timeofday'], ascending=[True, False])


# ## ********************

# Now, from the made train_data, we now added the dates which were not present.
# #### We included the days when no trade was done for a given bond. But assuring that on those days, one of the bond was traded. This improved the accuracy.
# #### This clearly shows that we have to take into account for each bond when they were not traded, as there is a high possibility for some bonds that they will not be traded on 10th, 13th and 14th June. 

# In[75]:

mn = main_isin_df["date"].max()
ep_to_day(mn)
#datetime.fromtimestamp(mn)


# In[78]:

print(main_isin_df["date"].nunique())
poss_dates = list(main_isin_df["date"].unique())
#print(poss_dates)


# In[13]:

len_tiger = len(main_isin_df_group)
main_isin_list = []
print("start")
side_list = [0, 0, 1, 1]
time_list = [0, 1, 0, 1]
full_list_bond_df = []
for i in range(len_tiger):
    if i%10000 == 0:
        print(i)
    main_per_bondname = main_isin_df_group[i][0]
    main_per_bond = main_isin_df_group[i][1]
    weekday_l = main_per_bond["weekday"].iloc[0]
    date_list = list(main_per_bond["date"].unique())
    list_for_each_bond = []
    list_for_each_bond.append(main_per_bond)
    for x in poss_dates:
        if x not in date_list:
            tmp_df = pd.DataFrame(index = np.arange(4))
            tmp_df["isin"] = main_per_bondname
            tmp_df["date"] = x
            tmp_df["side"] = side_list
            tmp_df["timeofday"] = time_list
            tmp_df["volume"] = 0
            tmp_df["weekday"] = weekday_l
            list_for_each_bond.append(tmp_df)
    for_each_bond_df = pd.concat(list_for_each_bond)
    full_list_bond_df.append(for_each_bond_df)



# In[20]:

MAIN_df_now = pd.concat(full_list_bond_df)


# In[21]:

MAIN_df_now.to_csv("MAIN_DF.csv")


# In[22]:

MAIN_df_now.shape


# In[23]:

MAIN_df_now.head()


# In[84]:

MAIN_df_now.shape


# In[85]:

MAIN_df_now.describe()


# In[88]:

MAIN_df_now["date"].max() - MAIN_df_now["date"].min()


# In[89]:

min_date = MAIN_df_now["date"].min()
again_poss_dates = list(MAIN_df_now["date"].unique())


# In[97]:

print(min_date)
MAIN_df_now["date"].max()


# In[90]:

# MAIN_GROUP = list(MAIN_df_now.groupby("isin"))


# In[105]:

# no_date = 1464546600


#  ### <<< -------------- 
#  ---- This was for adding the "no_date" date, i.e., "29th May" to the dataset. 
#  In this day none of the bonds were traded.
#  #### But adding this and then then combining with the existing framework decreased the efficiency of the model.
#  So, commenting the entire section for this.
#  

# In[111]:


"""
len_tiger = len(MAIN_GROUP)
main_isin_list = []
print("start")
side_list = [0, 0, 1, 1]
time_list = [0, 1, 0, 1]
full_list_bond_df = []
for i in range(len_tiger):
    if i%10000 == 0:
        print(i)
    main_per_bondname = MAIN_GROUP[i][0]
    main_per_bond = MAIN_GROUP[i][1]
    weekday_l = main_per_bond["weekday"].iloc[0]
    # date_list = list(main_per_bond["date"].unique())
    list_for_each_bond = []
    list_for_each_bond.append(main_per_bond)
  
    tmp_df = pd.DataFrame(index = np.arange(4))
    tmp_df["isin"] = main_per_bondname
    tmp_df["date"] = no_date
    tmp_df["side"] = side_list
    tmp_df["timeofday"] = time_list
    tmp_df["volume"] = 0
    tmp_df["weekday"] = weekday_l
    list_for_each_bond.append(tmp_df)
    for_each_bond_df = pd.concat(list_for_each_bond)
    full_list_bond_df.append(for_each_bond_df)

"""


# In[112]:

# MMMAIN_df_now = pd.concat(full_list_bond_df)


# In[113]:

# print(MMMAIN_df_now.shape)
# MMMAIN_df_now.to_csv("FINAL_TRAIN_DATA_PART1.csv")


# In[15]:

#MMMAIN_df_now = pd.read_csv("FINAL_TRAIN_DATA_PART1.csv")


# In[16]:

#MMMAIN_df_now = MMMAIN_df_now.drop("Unnamed: 0", axis = 1)


# ###  ----------- >>>

# ### Again we saved the created MAIN_df_now dataframe to MAIN_DF.csv
# ### For multiple trials, we directly read it from the generated file. (If you are trying to run this notebook, just comment the next block.)
# (You can check that the code for generation of this file is above.)

# In[80]:

MAIN_df_now = pd.read_csv("MAIN_DF.csv")


# In[81]:

MAIN_df_now.columns.values


# In[82]:

MAIN_df_now = MAIN_df_now.drop("Unnamed: 0", axis = 1)


# ## ------------------------------------

# ### Now cleaning and pre processing the meta data

# In[17]:

#meta_data.isnull().sum()


# In[18]:

meta_data["issue date"].unique()


# In[19]:

meta_data["issuer"].nunique()


# In[20]:


price_feature = list(data.groupby("isin"))


# In[21]:

price_feature[0][1]["price"].describe()


# In[22]:

price_feature[1][1].shape


# In[23]:

p_tmp = price_feature[5][1].copy()
p_series = convert_epoch(p_tmp["date"])


# In[24]:

p_series.max()


# In[25]:

p_tmp1 = price_feature[5][1].copy()
aq = list(p_tmp1.groupby(["side"]))
aq[1][1]


# In[26]:

meta_data_1 = meta_data_clean1(meta_data)
meta_data_1.head()


# In[27]:

meta_data_2 = meta_data_clean2(meta_data_1)


# In[30]:

meta_data_2["issuer"].nunique()


# In[31]:

epoch_test = datetime.strptime("10Jun2016", '%d%b%Y')
epoch_test = epoch_test.timestamp()
def time_last_delta(t1):
    x = datetime.fromtimestamp(epoch_test)
    y = datetime.fromtimestamp(t1)
    z = x - y
    return z.days


# #### Adding features like mean_price, standard_deviation_price, number of days bond is traded, total price sell, total price buy.
# 

# In[32]:

map_mean = {}
map_std = {}
map_numdays = {}
map_day_std = {}
map_sell_num = {}
map_buy_num = {}

map_buy_vol = {}
map_sell_vol = {}

for i in range(meta_data.shape[0]):
    bond_num = price_feature[i][0]
    meanx = price_feature[i][1]["price"].mean()
    stdx = price_feature[i][1]["price"].std()
    
    # vol_mean = price_feature[i][1]["volume"].sum()
    map_mean[bond_num] = meanx
    map_std[bond_num] = stdx
    map_numdays[bond_num] = price_feature[i][1].shape[0]
    
    p_series = convert_epoch(price_feature[i][1]["date"])
    last_epoch1 = p_series.std()
    #print(last_epoch1)
    map_day_std[bond_num] = last_epoch1
    
    p_tmp1 = price_feature[i][1].copy()
    p_tmp_group = list(p_tmp1.groupby("side"))
    #print(p_tmp_group)
    if len(p_tmp_group) == 2:
        map_sell_num[bond_num] = p_tmp_group[1][1].shape[0]
        cat_sell = p_tmp_group[1][1]["volume"].sum()
        map_buy_num[bond_num] = p_tmp_group[0][1].shape[0]
        cat_buy = p_tmp_group[0][1]["volume"].sum()
    else:
        if p_tmp_group[0][0] == 0:
            map_buy_num[bond_num] = p_tmp_group[0][1].shape[0]
            map_sell_num[bond_num] = 0
            cat_buy = p_tmp_group[0][1]["volume"].sum()
            cat_sell = 0
        else:
            map_sell_num[bond_num] = p_tmp_group[0][1].shape[0]
            map_buy_num[bond_num] = 0
            cat_sell = p_tmp_group[0][1]["volume"].sum()
            cat_buy = 0
            
    map_buy_vol[bond_num] = meanx * cat_buy
    map_sell_vol[bond_num] = meanx * cat_sell

mean_col = []
std_col = []
trade_days_col = []
days_std_col = []

days_sell = []
days_buy = []

cat_sell = []
cat_buy = []

for bond in meta_data_2["isin"]:
    mean_col.append(map_mean[bond])
    std_col.append(map_std[bond])
    trade_days_col.append(map_numdays[bond])
    days_std_col.append(map_day_std[bond])
    days_sell.append(map_sell_num[bond])
    days_buy.append(map_buy_num[bond])
    cat_sell.append(map_sell_vol[bond])
    cat_buy.append(map_buy_vol[bond])


# In[ ]:




# In[33]:

meta_data_2["price_mean"] = mean_col
meta_data_2["price_std"] = std_col
meta_data_2["trade_days"] = trade_days_col

meta_data_2["cat_sell"] = cat_sell
meta_data_2["cat_buy"] = cat_buy
# meta_data_2["days_std"] = days_std_col
# meta_data_2["buy_days"] = days_buy
# meta_data_2["sell_days"] = days_sell
meta_data_2_tmp = meta_data_2.copy()


# In[38]:

meta_data_2.to_csv("meta_wihout_mat.csv",index=False)


# ## ------------------------------------------------

# ### Merging the data and the metadata together now

# In[83]:

mrg_data = MAIN_df_now.merge(meta_data_2,on='isin')


# In[84]:

mrg_data.head()


# In[85]:

mrg_data["diff_issue_date"] = mrg_data["date"] - mrg_data["issue date"] 
mrg_data = mrg_data.drop("issue date", axis = 1)
mrg_data.head()


# In[86]:

mrg_data.shape


# In[87]:

train_Y = mrg_data["volume"]
drop_col = ["isin", "volume"]
train_X = mrg_data.drop(drop_col, axis = 1)


# ## -----------------------------------------------------------------

# ### Creating the test dataset for 10th, 13th, 14th June.

# In[42]:

dates = []
date_epoch = []
dates.append("10Jun2016")
dates.append("13Jun2016")
dates.append("14Jun2016")
for x in dates:
    epoch = datetime.strptime(x, '%d%b%Y')
    date_epoch.append(epoch.timestamp())
date_epoch = date_epoch + date_epoch
date_epoch = date_epoch + date_epoch
side = [0, 0, 0, 1, 1, 1]
side = side + side
timeofday = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
len(date_epoch)


# In[43]:

bond_num = list(meta_data["isin"])
head_tmp = ["isin", "date", "side", "timeofday"]
df_list = []
for x in bond_num:
    tmp_df = pd.DataFrame(index = np.arange(12))
    tmp_df["isin"] = x
    tmp_df["date"] = date_epoch
    tmp_df["side"] = side
    tmp_df["timeofday"] = timeofday
    df_list.append(tmp_df)
main_df = pd.concat(df_list)
test = main_df


# In[44]:

le = LabelEncoder()
test["weekday"] = test["date"].apply(lambda x: ep_to_day(x))
test["weekday"] = le.fit_transform(test["weekday"])


# In[45]:

mrg_test_data = test.merge(meta_data_2_tmp,on='isin')
mrg_test_data["diff_issue_date"] = mrg_test_data["date"] - mrg_test_data["issue date"] 
mrg_test_data = mrg_test_data.drop("issue date", axis = 1)
mrg_test_data.head()


# In[46]:

test_X = mrg_test_data.drop("isin", axis = 1)


# In[47]:

test_X.columns.values


# ## --------------------------------------------------

# ### XGBoost Model

# In[88]:

ypred = run_xgboost(train_X, train_Y, test_X)


# ### test.head()

# In[89]:

ypred_pos = ypred.copy()
ypred_pos[0] = ypred_pos[0].apply(lambda x: 0.0 if x<0 else x)


# In[90]:

new_test_rew = test.copy()
new_test_rew.shape
ypred.shape
new_test_rew["volume"] = ypred_pos[0].values


# In[91]:

ypred[0].nunique()


# In[92]:

new_test_rew["volume"].unique()


# In[93]:

ftd_group_bond = list(new_test_rew.groupby("isin"))


# In[94]:

ftd_group_bond_side = list(ftd_group_bond[0][1].groupby("side"))


# In[95]:

ftd_group_bond_side[1][1]["isin"]


# ## -------------------------------------------

# ### The model predicts for each day and each timeofday for sell and buy differently. So, summing it up and making it into the required format.

# In[96]:

new_test_rew.shape
len(ftd_group_bond)


# In[97]:

test_qwerty = pd.DataFrame(index = np.arange(17261))


# In[98]:

col1 = []
col2 = []
col3 = []


# In[99]:

for i in range(17261):
    bond_num = ftd_group_bond[i][0]
    ftd_group_bond_side = list(ftd_group_bond[i][1].groupby("side"))
    t_buy = ftd_group_bond_side[0][1] #Buy
    t_sell = ftd_group_bond_side[1][1] #Sell
    buy_vol = t_buy["volume"].sum()
    sell_vol = t_sell["volume"].sum()
    col1.append(bond_num)
    col2.append(buy_vol)
    col3.append(sell_vol)


# In[100]:

len(col3)


# In[101]:

test_qwerty["isin"] = col1
test_qwerty["buyvolume"] = col2
test_qwerty["sellvolume"] = col3


# In[102]:

test_qwerty.head()


# In[103]:

test_qwerty["buyvolume"].unique()


# In[104]:

test_qwerty.to_csv("SOL_MAIN_TRY_LION.csv", index = False)


# In[105]:

ypred[0].nunique()


# In[ ]:



