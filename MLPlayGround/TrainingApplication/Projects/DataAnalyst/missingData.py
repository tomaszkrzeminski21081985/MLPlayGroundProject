
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import  seaborn as sns


# series = pd.Series([1,2,3,4,np.NaN,6,7,8,9,None])


# print(series.isna())



# series[series.isna()]=9

# print(series)


# series_not_nan = series[series.notna()]

# print(series_not_nan)


# data = pd.DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan],
#                      [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])
# print(data)


# print(data.dropna(axis="columns",how="any"))

# data.iloc[:,0]=6

# print(data.dropna(axis="columns",how="any"))


# data = pd.DataFrame({"k1": ["one", "two"] * 3 + ["two"],
#                      "k2": [1, 1, 2, 3, 3, 4, 5]})

# print(data.duplicated())



# ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]

# bins = [18, 25, 35, 60, 100]
# age_categories = pd.cut(ages, bins)

# print(age_categories)



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Dane przyk³adowe
data = {"Kategoria": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"],
        "Warto": [10, 20, 30, 40, 15, 25, 35, 45, 12, 22, 32, 42, 18, 28, 38, 48]}

# Tworzenie DataFrame
df = pd.DataFrame(data)

# Ustawienie catplot
sns.catplot(x="Kategoria", y="Warto", data=df, kind="bar" )
sns.catplot(x="Kategoria", y="Warto", data=df, kind="bar" )
sns.catplot(x="Kategoria", y="Warto", data=df, kind="boxen" )
sns.catplot(x="Kategoria", y="Warto", data=df, kind="box" )

# Poka¿ wykres
plt.show()
# df = pd.concat([df] * 3, ignore_index=True)

# # Wyœwietlenie DataFrame
# print(df)


# names=df.iloc[:,0].unique()

# print(names)

# df_grouped = df.groupby('name')[['bonus', 'bonus_value']].sum().reset_index()

# # Utwórz now¹ kolumnê, która odejmuje bonus_value od bonus
# df_grouped['diff'] = df_grouped['bonus'] - df_grouped['bonus_value']

# print(df_grouped)


# arr = np.arange(12).reshape(3, 4)

# t = np.concatenate([arr, arr], axis=1)

# print(t)


# t2 = np.concatenate([arr, arr], axis=0)
# print(t2)




# data=np.arange(100)
# print(data)


# plt.plot(data)

# plt.show()


# fig, axes = plt.subplots(2, 2)

# x = [1, 2, 3, 8, 5]
# y = [2, 6, 6, 8, 10]

# axes[0,0].scatter(x, y, color="red", linestyle="dashed", marker="o", label="steps-post")
# axes[0,0].set_xticks([2,4,8,10])
# axes[0,0].set_xlabel(["dwa","cztery","osiem","dziesi"], fontsize=10)

# axes[0,1].plot(np.arange(10, 100), np.arange(10, 100), color="red", linestyle="dashed")
# data = np.concatenate((np.arange(1, 41), np.arange(30, 70), np.arange(10, 20)))
# axes[1,0].hist(data, bins=10, color="blue", alpha=1)

# axes[1,1].plot(np.arange(10), label="steps-post-1", color="red")
# axes[1,1].plot(np.arange(20), label="steps-post-2", color="blue")
# axes[1,1].plot(np.arange(40), label="steps-post-3", color="yellow")

# axes[0,0].set_title("raz")
# axes[0,1].set_title("dwa")
# axes[1,0].set_title("trzy")
# axes[1,1].set_title("czter")

# axes[1,1].legend()

# plt.show()

# fig.savefig(r"C:\Users\tomas\OneDrive\Pulpit\wykres.png")


# from datetime import datetime

# fig, ax = plt.subplots()

# with open("przyklady/spx.csv", encoding="utf-8") as f:
#     data = pd.read_csv(f, index_col=0, parse_dates=True)
# spx = data["SPX"]

# spx.plot(ax=ax, color="black")

# crisis_data = [
#     (datetime(2007, 10, 11), "Szczyt hossy"),
#     (datetime(2008, 3, 12), "Upadek Bear Stearns"),
#     (datetime(2008, 9, 15), "Bankructwo Lehman")
# ]

# for date, label in crisis_data:
#     ax.annotate(label, xy=(date, spx.asof(date) + 75),
#                 xytext=(date, spx.asof(date) + 225),
#                 arrowprops=dict(facecolor="black", headwidth=4, width=2,
#                                 headlength=4),
#                 horizontalalignment="left", verticalalignment="top")

# # Przedzia³ lat 2007-2010
# ax.set_xlim(["1/1/2007", "1/1/2011"])
# ax.set_ylim([600, 1800])
# ax.set_xlabel("Data")

# ax.set_title("Wazne wydarzenia kryzysu finansowego 2008-2009")

fig,axes=plt.subplots(2,2)

data =pd.Series(np.random.uniform(size=16),index=list("abcdefghijklmnop"))

data.plot.bar(ax=axes[0,0], color="red", alpha=0.5)
data.plot.barh(ax=axes[1,0], color="blue", alpha=0.5)

plt.show()