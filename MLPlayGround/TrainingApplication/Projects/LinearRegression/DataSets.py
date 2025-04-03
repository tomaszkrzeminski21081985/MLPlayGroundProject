from tkinter import Grid
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from Features import GetData
from Evaluation import CheckLinerRegression, GridSearch_Evaluation, GridSearchRandom_Evaluation        


getData=GetData()
housing=getData.load_housing_data()
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

print(housing.describe())

strat_train_set, strat_test_set = train_test_split(
housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

preprocessing = getData.GetPreprocessing()

linear_reg = CheckLinerRegression(housing, housing_labels)
linear_reg.Run(preprocessing)

######################### GridSearchCV #########################

tree_req=make_pipeline(preprocessing,DecisionTreeRegressor(random_state=42))
tree_req.fit(housing,housing_labels)
housing_predictions=tree_req.predict(housing)
tree_rsme=-cross_val_score(tree_req,housing,housing_labels,scoring="neg_root_mean_squared_error",cv=3)
print(pd.Series(tree_rsme).describe())

forest_reg=make_pipeline(preprocessing,RandomForestRegressor(random_state=42))
forest_rmse=cross_val_score(forest_reg,housing,housing_labels,scoring="neg_root_mean_squared_error",cv=3)
print(pd.Series(forest_rmse).describe())


grid_search_eval = GridSearch_Evaluation(housing,housing_labels)

grid_search_eval.SearchForBestParameters(preprocessing)


grid_searchRandom_eval = GridSearchRandom_Evaluation(housing,housing_labels)

grid_searchRandom_eval.SearchForBestParameters(preprocessing)

