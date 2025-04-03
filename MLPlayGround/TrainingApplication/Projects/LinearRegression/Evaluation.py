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
from sklearn.metrics import mean_absolute_error


class GridSearch_Evaluation:

    def __init__(self,housing,housing_labels):
         print('GridSearch_Evaluation')
         # strat_train_set, strat_test_set = train_test_split(
         # data, test_size=0.2, stratify=data["income_cat"], random_state=42)

         # self.housing = strat_train_set.drop("median_house_value", axis=1)
         # self.housing_labels = strat_train_set["median_house_value"].copy()
         self.housing=housing
         self.housing_labels=housing_labels
         

    def SearchForBestParameters(self,preprocessing):
        full_pipeline = Pipeline([
             ("preprocessing", preprocessing),
             ("random_forest", RandomForestRegressor(random_state=42)),
             ])

        param_grid = [
              {'preprocessing__geo__n_clusters': [5, 8, 10],
                 'random_forest__max_features': [4, 6, 8]},
                {'preprocessing__geo__n_clusters': [10, 15],
                 'random_forest__max_features': [6, 8, 10]},
            ]
        grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                                       scoring='neg_root_mean_squared_error')
        grid_search.fit(self.housing, self.housing_labels)

        cv_res = pd.DataFrame(grid_search.cv_results_)        

        cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
        print(cv_res)



class GridSearchRandom_Evaluation:

    def __init__(self,housing,housing_labels):
         print('GridSearchRandom_Evaluation')         
         self.housing=housing
         self.housing_labels=housing_labels

    def SearchForBestParameters(self,preprocessing):
        full_pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("random_forest", RandomForestRegressor(random_state=42)),
])
        param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

        rnd_search = RandomizedSearchCV(
            full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
            scoring='neg_root_mean_squared_error', random_state=42)

        rnd_search.fit(self.housing, self.housing_labels)

        cv_res = pd.DataFrame(rnd_search.cv_results_)
        cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
        print(cv_res)


        # poka¿ najlepszy model
        final_model=rnd_search.best_estimator_
        feature_importances = final_model["random_forest"].feature_importances_
        feature_importances.round(2)
        print(sorted(zip(feature_importances,
                   final_model["preprocessing"].get_feature_names_out()),
                   reverse=True))


class CheckLinerRegression:

    def __init__(self,housing,housing_labels):
        print("LinerRegression")
        self.housing=housing
        self.housing_labels=housing_labels


    def Run(self,preprocessing):
        lin_req = make_pipeline(preprocessing, LinearRegression())
        lin_req.fit(self.housing, self.housing_labels)

        housing_predictions = lin_req.predict(self.housing)
        # print(housing_predictions[0:5].round(-2))
        # print(housing_labels.iloc[:5].values)       

        lin_rsme = mean_absolute_error(self.housing_labels, housing_predictions)
        print(lin_rsme)
