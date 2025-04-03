from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

class CheckLinerRegression:

    def __init__(self,data,data_labels,data_test,labels_test):
        print("Liner Regression")
        self.data=data
        self.data_labels=data_labels
        self.data_test=data_test
        self.labels_test=labels_test


    def Run(self,preprocessing):
        lin_req = make_pipeline(preprocessing, LinearRegression())
        lin_req.fit(self.data, self.data_labels)

        data_predictions = lin_req.predict(self.data_test)
        predictions_df = pd.DataFrame({
        'Aktualna': self.labels_test.iloc[:20].values,
        'Przewidywana': data_predictions[0:20].round(-2)
        })

        print(predictions_df)

        lin_rsme = mean_absolute_error(self.labels_test, data_predictions)
        print(lin_rsme)        

        coefficients = pd.DataFrame(lin_req.named_steps['linearregression'].coef_, self.data.columns)

        print(coefficients)

        print("end")