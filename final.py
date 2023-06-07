import pandas as pd
import numpy as np


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf['price per sqft'])
        st = np.std(subdf['price per sqft'])
        reduced_df = subdf[(subdf['price per sqft']>(m-st)) & (subdf['price per sqft']<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price per sqft']),
                'std': np.std(bhk_df['price per sqft']),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price per sqft']<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')


df = pd.read_csv("Bengaluru_House_Data.csv")
df2 = df.drop(['area_type','availability','society','balcony'],axis = 1)
df3 = df2.dropna()

df3['bhk'] = df3['size'].apply(lambda x: int(x.split()[0]))
df3 = df3.drop(['size'],axis=1)

df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]

df5 = df4.copy()
df5['price per sqft'] = round(df5['price']*100000/df5['total_sqft'],3)


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)

loc_stats_less = location_stats[location_stats<=10]
df5['location'] = df5['location'].apply(lambda x: 'other' if x in loc_stats_less else x)

df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df7 = remove_pps_outliers(df6)
df8 = remove_bhk_outliers(df7)

df9 = df8[df8.bath<df8.bhk+2]
df10 = df9.drop('price per sqft',axis=1)

dum = pd.get_dummies(df10.location)
df11 = pd.concat([df10,dum.drop('other',axis=1)],axis = 'columns')

df12 = df11.drop('location',axis = 'columns')

#model
x = df12.drop('price',axis=1)
y = df12.price

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=10)

from sklearn.linear_model import LinearRegression
#reg classifier
lr_clf= LinearRegression()
lr_clf.fit(x_train.values,y_train.values)

X=x
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

#print(predict_price('Indira Nagar',1000,3,3))
#aww yiss


#export artifacts
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
