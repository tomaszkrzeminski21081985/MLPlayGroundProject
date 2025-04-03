from Features import GetData
from Evaluation import CheckLinerRegression
from sklearn.model_selection import train_test_split

data=GetData()
df=data.load_data()


# data.look_for_Correlations()
df = df.drop(['Email', 'Address', 'Avatar'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']],
    df['Yearly Amount Spent'],
    test_size=0.2,
    # stratify=df["Avg. Session Length"],
    random_state=42)

print(X_train.info())
print(y_train.info())

preprocessing = data.GetPreprocessing(X_train)
linear_reg =CheckLinerRegression(X_train, y_train,X_test,y_test)
linear_reg.Run(preprocessing)   