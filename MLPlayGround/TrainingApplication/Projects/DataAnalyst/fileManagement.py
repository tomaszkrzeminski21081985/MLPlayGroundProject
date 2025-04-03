

from math import nan
import pandas as pd

pd.options.display.min_rows=100

df = pd.read_csv('C:\\Users\\tomas\\source\\repos\\MLPlayGround\\TrainingApplication\\Projects\\DataAnalyst\\examples\\akcje.csv',parse_dates=True, skiprows=[0], names=['a', 'b', 'c', 'd', 'e'])
print(df)

df.iloc[0,1]=nan


df.iloc[10,2]=nan


df['b'] = df['b'].fillna(0)
df['c'] = df['c'].fillna(0)
df['d'] = df['d'].fillna(0)
df['e'] = df['e'].fillna(0)


print(df.isna())

print("Null counts:")
print(df.isnull().sum().to_string())

print("NaN counts:")
print(df.isna().sum().to_string())  # Note: to_string() instead of tostring()

print("Column types:")
print(df.dtypes)  # Note: dtypes instead of types

print("Dataframe info:")
print(df.info())  # Note: info() is a method, so you need to call it with ()