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




class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Podobienstwo {i} skupienia" for i in range(self.n_clusters)]

class  GetData:
    
    def column_ratio(self,X):
       return X[:, [0]] / X[:, [1]]

    def ratio_name(self,function_transformer, feature_names_in):
        return ["ratio"]  

    def ratio_pipeline(self):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(self.column_ratio, feature_names_out=self.ratio_name),
            StandardScaler())

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    def __init__(self):
        print('start')
        

       


    def Describe(self, frame):
            print(frame.head()) 
            print(frame.info())
            print(frame.describe())
            print(frame["ocean_proximity"].describe())
            print(frame["ocean_proximity"].value_counts())

    def load_housing_data(self):
        current_file = Path(__file__)
        traball_path = current_file.parent / "housing.tgz"
        datasets_path = current_file.parent / "datasets"

        if not traball_path.is_file():
            datasets_path.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/ageron/data/raw/main/housing.tgz"
            urllib.request.urlretrieve(url, traball_path)
            with tarfile.open(traball_path) as housing_tarball:
                housing_tarball.extractall(path=datasets_path)
        data= pd.read_csv(datasets_path / "housing" / "housing.csv")        
        return data


    def GetPreprocessing(self):
        log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
            FunctionTransformer(np.log, feature_names_out="one-to-one"),
            StandardScaler())
        cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
        default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

        preprocessing = ColumnTransformer([
        ("wspolczynnik_sypialni", self.ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("pokoje_na_rodzine", self.ratio_pipeline(), ["total_rooms", "households"]),
        ("liczba_osob_na_dom", self.ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", self.cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline)
        
        return preprocessing

    def  CreateHistogram(self,frame,b,f=(12,8)):
        frame.hist(bins=b,figsize=f) 
        plt.show()

    # def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    #     path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    #     if tight_layout:
    #         plt.tight_layout()
    #     plt.savefig(path, format=fig_extension, dpi=resolution)

    # def ModifyData(self,data):
    #     data["income_cat"] = pd.cut(data["median_income"],
    #                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    #                            labels=[1, 2, 3, 4, 5])

    #     imputer=SimpleImputer(strategy="median")
    #     housing_num=data.select_dtypes(include=[np.number])
    #     imputer.fit(housing_num)
    #     X = imputer.transform(housing_num)
    #     housing_tr = pd.DataFrame(X, columns=housing_num.columns,
    #                       index=housing_num.index)



    def Shufle_And_SplitData(self,data,test_ratio):
        shuffled=np.random.permutation(len(data))
        test_set_size=int(len(data)*test_ratio)
        test_indices=shuffled[:test_set_size]
        train_indices=shuffled[test_set_size:]
        return data.iloc[train_indices],data.iloc[test_indices]
    
    # def PlotBox(self, frame):
    #     columns_to_plot = [col for col in frame.columns if col != "ocean_proximity"]
    #     plt.boxplot([frame[col] for col in columns_to_plot])
    #     plt.title('Wykres pude³kowy cech dataframe')
    #     plt.xlabel('Cecha')
    #     plt.ylabel('Wartoœæ')
    #     plt.xticks(range(1, len(columns_to_plot) + 1), columns_to_plot)
    #     plt.show()
        
    def ShowData(self,data):
        print()
        # print(len(strat_test_set))
        # print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
        # housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
        #      s=housing["population"] / 100, label="populacja",
        #      c="median_house_value", cmap="jet", colorbar=True,
        #      legend=True, sharex=False, figsize=(10, 7))

        # plt.show()

    def LookForCorelation(sefl,data):
        corr_matrix = data.loc[:, data.columns != 'ocean_proximity'].corr()
        print(corr_matrix["median_house_value"].sort_values(ascending=False) )   
        attributes= ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
        scatter_matrix(data[attributes],figsize=(12,8))
        plt.show