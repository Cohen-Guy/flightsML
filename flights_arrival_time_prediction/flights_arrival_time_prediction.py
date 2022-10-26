import os
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from globalsContext import GlobalsContextClass
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from hyper_parameters_tunning import HyperParametersTunning
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

class FlightsArrivalTimePrediction:

    def __init__(self):
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.globals_context = GlobalsContextClass(time_str)
        self.debug_flag = True
        dataset_csv_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'flights_arrival_time_prediction', 'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2021_1.csv')
        self.dataset = pd.read_csv(dataset_csv_file_path)
        self.target_col_name = 'ArrDelay'
        self.target_bucket_col_name = 'DelayBucket'
        self.target_ordinal_bucket_col_name = 'DelayBucketOrdinal'
        self.boolean_cols = [boolean_col['field_name'] for boolean_col in self.globals_context.cols_dict['boolean_cols'] if not boolean_col['exclude_feature_from_training']]
        self.ordinal_cols = [ordinal_col['field_name'] for ordinal_col in self.globals_context.cols_dict['ordinal_cols'] if not ordinal_col['exclude_feature_from_training']]
        self.categorical_cols = [categorical_col['field_name'] for categorical_col in self.globals_context.cols_dict['categorical_cols'] if not categorical_col['exclude_feature_from_training']]
        self.numerical_cols = [numerical_col['field_name'] for numerical_col in self.globals_context.cols_dict['numerical_cols'] if not numerical_col['exclude_feature_from_training']]
        self.datetime_cols = [datetime_col['field_name'] for datetime_col in self.globals_context.cols_dict['datetime_cols'] if not datetime_col['exclude_feature_from_training']]
        self.selected_columns = self.boolean_cols + self.ordinal_cols + self.categorical_cols + self.numerical_cols + self.datetime_cols

    def dataset_extract_target(self, dataset):
        dataset = self.globals_context.divide_target_to_buckets(dataset, self.target_col_name, self.target_bucket_col_name)
        dataset = self.globals_context.transform_delay_bucket_to_ordinal(dataset)
        dataset.pop(self.target_bucket_col_name)
        y = dataset.pop(self.target_ordinal_bucket_col_name)
        return dataset, y

    def feature_selection(self, X):
        # self.selected_features = ['origin_location_code', 'destination_location_code', 'departure_date', 'validatingAirlineCodes', 'distance']  # 'validatingAirlineCodes',
        # self.excluded_features = ['FlightDate']
        # X = X.loc[:, ~X.columns.isin(self.excluded_features)]
        X = X[self.selected_columns]
        return X



    def outlier_detection(self, X, excluded_column):
        isolation_forest_model = IsolationForest(max_samples=100, random_state=42)
        X_without_excluded_columns = X.loc[:, X.columns != excluded_column]
        X['outlier'] = isolation_forest_model.fit_predict(X_without_excluded_columns)
        return X

    def encoding(self, X, y):
        target_encoder = TargetEncoder()
        X[self.categorical_cols] = target_encoder.fit_transform(X[self.categorical_cols], y)
        return X, y

    def scaling(self, X):
        scaler = StandardScaler()
        X[self.numerical_cols] = scaler.fit_transform(X[self.numerical_cols])
        return X
    def cleaning(self, dataset):
        # dataset = dataset[dataset[self.target_col_name].notna()]
        dataset.fillna(0, inplace=True)
        return dataset
    def preprocessing(self, dataset):
        # predict_for_airline_code = 'AF'
        # dataset = self.filter_dataset_by_airline_code(dataset, predict_for_airline_code)
        dataset = self.cleaning(dataset)
        dataset = self.transform_types(dataset)
        X, y = self.dataset_extract_target(dataset)
        X = self.feature_selection(X)
        X = self.globals_context.feature_engineering(X, False, False, False)
        # X = self.scaling(X)
        X, y = self.encoding(X, y)
        # excluded_column = 'departure_date'
        # X = self.outlier_detection(X, excluded_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        return X_train, X_test, y_train, y_test

    def transform_types(self, dataset):
        dataset[self.categorical_cols] = dataset[self.categorical_cols].astype(str)
        return dataset

    # def train_linear_regression(self, X_train, y_train):
    #     model = LinearRegression()
    #     model.fit(X_train, y_train)
    #     return model

    def train_xgboost_classifier(self, X_train, y_train):
        params = {
            'booster': 'dart',
            'lambda': 0.5642581562857996,
                'alpha': 0.011964958656080679,
        'subsample': 0.9664606774005825,
        'colsample_bytree': 0.973523784845717,
        'max_depth': 9,
        'min_child_weight': 3,
        'eta': 0.9741117119835012,
        'gamma': 0.00015275824324089936,
        'grow_policy': 'depthwise',
        'sample_type': 'uniform',
        'normalize_type': 'tree',
        'rate_drop': 3.2087168422238625e-05,
        'skip_drop': 3.4557151831426346e-05,
        }
        model = XGBClassifier(params)
        model.fit(X_train, y_train)
        return model

    def train_logistic_regression(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def explainability(self, model, X, y):
        shap_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'explainability', 'shap')
        sample_ind = 20
        sampled_x = X.sample(frac=0.1)
        explainer = shap.Explainer(model, sampled_x)
        shap_values = explainer(sampled_x)
        waterfall_diagram = shap.plots.waterfall(shap_values[0], show=False)
        waterfall_diagram_name = f"waterfall_diagram_{self.time_str}.png"
        waterfall_diagram_path = os.path.join(shap_folder_path, waterfall_diagram_name)
        waterfall_diagram.savefig(waterfall_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()
        shap.plots.beeswarm(shap_values, show=False)
        beeswarm_diagram_name = f"beeswarm_diagram_{self.time_str}.png"
        beeswarm_diagram_path = os.path.join(shap_folder_path, beeswarm_diagram_name)
        beeswarm_diagram = plt.gcf()
        beeswarm_diagram.savefig(beeswarm_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()
        # summary_plot_fig = shap.summary_plot(shap_values[0], X_test, show=False)
        # summary_plot_diagram_name = f"summary_plot_diagram_{self.time_str}.png"
        # summary_plot_path = os.path.join(summary_plot_fig, summary_plot_diagram_name)
        # summary_plot_fig.savefig(summary_plot_path, format='png', dpi=600, bbox_inches='tight')

        # for feature in self.selected_features:
        feature = 'distance'
        self.shap_plots_for_feature(model, feature, shap_values, shap_folder_path, sampled_x, sample_ind)
        # feature = 'departure_date'
        # self.shap_plots_for_feature(feature, shap_values, shap_folder_path, sampled_x, sample_ind)

    def shap_plots_for_feature(self, model, feature, shap_values, shap_folder_path, sampled_x, sample_ind):
        feature_shap_values = shap_values[:, feature]
        shap.plots.scatter(feature_shap_values, color=shap_values, show=False)
        scatter_diagram_name = f"{feature}_scatter_diagram_{self.time_str}.png"
        scatter_diagram_path = os.path.join(shap_folder_path, scatter_diagram_name)
        scatter_diagram = plt.gcf()
        scatter_diagram.savefig(scatter_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()
        partial_dependence_diagram, ax = shap.partial_dependence_plot(
            feature, model.predict, sampled_x, model_expected_value=True,
            feature_expected_value=True, ice=False,
            shap_values=shap_values[sample_ind:sample_ind + 1, :],
            show=False
        )
        partial_dependence_diagram_name = f"{feature}_partial_dependence_diagram_{self.time_str}.png"
        partial_dependence_diagram_path = os.path.join(shap_folder_path, partial_dependence_diagram_name)
        partial_dependence_diagram.savefig(partial_dependence_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()

    def train_random_forest_classifier(self, X_train, y_train):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model

    def evaluate_train(self, model, X, y):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        X['y_true'] = y
        X['y_pred'] = y_pred
        return X, accuracy

    def evaluate_test(self, model, X, y):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        X['y_true'] = y
        X['y_pred'] = y_pred
        return X, accuracy


    def ml_flow(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self.dataset)
        model = self.train_xgboost_classifier(X_train, y_train)
        X_train, train_accuracy = self.evaluate_train(model, X_train, y_train)
        print(f"train_accuracy: {train_accuracy}")
        self.confusion_matrix_plot(X_train['y_true'], X_train['y_pred'])
        X_test, test_accuracy = self.evaluate_test(model, X_test, y_test)
        print(f"test_accuracy: {test_accuracy}")
        self.confusion_matrix_plot(X_test['y_true'], X_test['y_pred'])
        return model, X_test, y_test

    def hyperparameters_optimization(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self.dataset)
        self.hyperParametersTunning = HyperParametersTunning(X_train, X_test, y_train, y_test)
        self.hyperParametersTunning.optimize()


    def confusion_matrix_plot(self, y_true, y_pred):
        delay_bucket_confusion_matrix = confusion_matrix(y_true, y_pred)
        x = ['{-10000, -60}', '{-60, -45}', '{-45, -30}', '{-30, -15}', '{-15, -3}', '{-3, 3}', '{3, 15}', '{15, 30}', '{30, 45}', '{45, 60}', '{60, 10000}']
        y = ['{-10000, -60}', '{-60, -45}', '{-45, -30}', '{-30, -15}', '{-15, -3}', '{-3, 3}', '{3, 15}', '{15, 30}', '{30, 45}', '{45, 60}', '{60, 10000}']

        # change each element of z to type string for annotations
        z_text = [[str(y) for y in x] for x in delay_bucket_confusion_matrix]

        # set up figure
        fig = ff.create_annotated_heatmap(delay_bucket_confusion_matrix, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

        # add title
        fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                          # xaxis = dict(title='x'),
                          # yaxis = dict(title='x')
                          )

        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))

        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=-0.35,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig['data'][0]['showscale'] = True
        fig.show()

if __name__ == "__main__":
    flights_arrival_time_prediction = FlightsArrivalTimePrediction()
    # flights_arrival_time_prediction.hyperparameters_optimization()
    model, X_test, y_test = flights_arrival_time_prediction.ml_flow()
    # fareMLPrediction.explainability(model, X_test, y_test)