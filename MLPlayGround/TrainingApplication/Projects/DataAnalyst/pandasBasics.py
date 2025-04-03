
import pandas as pd
import numpy as np






# table=["Jeden","Dwa","Trzy"]
# ser=pd.Series(table)

# print(ser)

# sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
# obj3 = pd.Series(sdata)

# states = ["California", "Ohio", "Oregon", "Texas"]
# obj4 = pd.Series(sdata, index=states)
# print(obj3 + obj4)

# data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada","Texas"],
#         "year": [2000, 2001, 2002, 2001, 2002, 2003,2004],
#         "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2,1]}
# frame = pd.DataFrame(data)




# frame["test"]=pd.Series([1,2],index=["Ohio","Nevada"])




# frame.drop(["test"],axis=1)

# print(frame)


# states2 = ["California", "Ohio","Ohio", "Oregon", "Texas"]
# obj4 = pd.Series(range(5), index=states2)


# print(obj4)


# table=pd.DataFrame(np.arange(25).reshape(5,5),index=["California", "Ohio","Ohio", "Oregon", "Texas"],columns=["a","b","c","d","e"])

# print(table)

# # print(table[table["a"]])


# print(table.iloc[0,0])

# print(table.iloc[4,4])

# print(table.iloc[[0,1]])

# print(table.iloc[[3,4]])

# print(table.loc["California", : ])

# print(table.loc[ : ,"a"])


# print(table["Ohio"])


# df1=pd.DataFrame(np.arange(12).reshape(3,4),columns=list('abcd'))
# df2=pd.DataFrame(np.arange(20).reshape(4,5),columns=list('abcde'))


# print(df1+df2)

# print('--------------')

# print(df1.add(df2,fill_value=0))


# print(df2.add(df1,fill_value=0))




table=pd.DataFrame(np.arange(12).reshape(4,3),index=["Utah", "Ohio","Texas", "Oregon"],columns=list('bde'))

print(table)

series=table.loc[ 'Utah']



print(table-series)


