import pandas as pd
from datetime import datetime

def load_data(path):
    return pd.read_csv(path)

def to_season(num):
    if num == 0:
        return 'spring'
    elif num == 1:
        return 'summer'
    elif num == 2:
        return 'fall'
    elif num == 3:
        return 'winter'
    return -1


def add_new_columns(df):
    df['season_name'] = df['season'].apply(to_season)
    df['Hour'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%H'))
    df['Day'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%d'))
    df['Month'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%m'))
    df['Year'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y'))
    df['is_weekend_holiday'] = df[['is_holiday','is_weekend']].apply(lambda x: 1 + 2*x['is_holiday']+x['is_weekend'], axis=1)
    df['t_diff'] = df[['t1','t2']].apply(lambda x: x['t1']-x['t2'], axis=1)
    return df


def data_analysis(df):
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()

    corrdict = {}
    for i in corr:
        for j in corr:
            if i != j and (j,i) not in corrdict:
                corrdict[(i,j)] = corr.loc[i,j]

    sorted_dict_ascending = dict(sorted(corrdict.items(), key=lambda item: abs(item[1])))[:5]
    sorted_dict_descending = dict(sorted(corrdict.items(), key=lambda item: abs(item[1]), reverse=True))[:5]

    print("Highest correlated values:")
    for i in range(5):
        key = list(sorted_dict_ascending.items())[i][0]
        value = list(sorted_dict_ascending.items())[i][1]
        print(f"{i + 1}. {key} with {value:%6f}")

    print("Lowest correlated values:")
    for i in range(5):
        key = list(sorted_dict_descending.items())[i][0]
        value = list(sorted_dict_descending.items())[i][1]
        print(f"{i + 1}. {key} with {value:%6f}")

