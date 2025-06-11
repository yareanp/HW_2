import pandas as pd
from datetime import datetime

def load_data(path):
    """reads and returns the pandas DataFrame"""
    return pd.read_csv(path)

def to_season(num):
    """returns a season name for each number between 0 and 3"""
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
    """"adds columns to df and returns the new df"""
    df['season_name'] = df['season'].apply(to_season)
    df['Hour'] = df['timestamp'].apply(lambda x: int(x[11:13]))
    df['Day'] = df['timestamp'].apply(lambda x: int(x[:2]))
    df['Month'] = df['timestamp'].apply(lambda x: int(x[3:5]))
    df['Year'] = df['timestamp'].apply(lambda x: int(x[6:10]))
    df['is_weekend_holiday'] = df[['is_holiday','is_weekend']].apply(lambda x: 1 + 2*x['is_holiday']+x['is_weekend'], axis=1)
    df['t_diff'] = df[['t1','t2']].apply(lambda x: x['t2']-x['t1'], axis=1)
    return df


def data_analysis(df):
    """prints statistics on the transformed df"""
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

    sorted_dict_ascending = dict(list(sorted(corrdict.items(), key=lambda item: abs(item[1])))[:5])
    sorted_dict_descending = dict(list(sorted(corrdict.items(), key=lambda item: abs(item[1]), reverse=True))[:5])

    print("Highest correlated are: ")
    for i in range(5):
        key = list(sorted_dict_descending.items())[i][0]
        value = list(sorted_dict_descending.items())[i][1]
        print(f"{i + 1}. {key} with {abs(value):.6f}")

    print("\nLowest correlated are: ")
    for i in range(5):
        key = list(sorted_dict_ascending.items())[i][0]
        value = list(sorted_dict_ascending.items())[i][1]
        print(f"{i + 1}. {key} with {abs(value):.6f}")
    print()


    season_means = df.groupby('season')['t_diff'].mean()
    print(f"fall average t_diff is {season_means.loc[2]:.2f}")
    print(f"spring average t_diff is {season_means.loc[0]:.2f}")
    print(f"summer average t_diff is {season_means.loc[1]:.2f}")
    print(f"winter average t_diff is {season_means.loc[3]:.2f}")
    print(f"All average t_diff is {df.mean(numeric_only=True)['t_diff']:.2f}")