import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector


class GetData:

    def __init__(self):
        print('Get Data Ecommerce Customers')
        self.Path = r'C:\Users\tomas\source\repos\MLPlayGround\TrainingApplication\Projects\DataAnalyst\examples\LinearRegression2\EcommerceCustomers'
        self.df = None

    def load_data(self):
        df = pd.read_csv(self.Path)
        print(df.describe())
        print(df.shape)
        print(df.info())
        self.df = df
        return df

    def look_for_Correlations(self):
        df_cor = self.df.iloc[:, 3:]

        print(df_cor.info())

        # zaokr¹glij wartoœci do 2 miejsc po przecinku
        print(df_cor.corr().round(2))
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_cor.corr(), annot=True, cmap='coolwarm', square=True)
        plt.title('Macierz Korelacji')
        plt.show()

    def GetPreprocessing(self,dataframe):
        log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
        
        preprocessing = ColumnTransformer([
            ("num", log_pipeline, dataframe.columns)
        ], remainder="passthrough")

        return preprocessing
