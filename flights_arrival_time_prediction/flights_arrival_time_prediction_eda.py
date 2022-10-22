import os
import pandas as pd
# import plotly.io as pio
import plotly.express as px
# import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
from globalsContext import GlobalsContextClass
import datetime

class FareMLEDA:

    def __init__(self):
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        pd.set_option('display.max_columns', None)
        self.debug_flag = True
        self.target_bucket_col_name = 'DelayBucket'
        self.target_ordinal_bucket_col_name = 'DelayBucketOrdinal'
        dataset_csv_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'flights_arrival_time_prediction', 'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2021_1.csv')
        self.dataset = pd.read_csv(dataset_csv_file_path)
        self.globals_context = GlobalsContextClass(time_str)
        self.boolean_cols = [boolean_col['field_name'] for boolean_col in self.globals_context.cols_dict['boolean_cols'] if not boolean_col['exclude_feature_from_training']]
        self.ordinal_cols = [ordinal_col['field_name'] for ordinal_col in self.globals_context.cols_dict['ordinal_cols'] if not ordinal_col['exclude_feature_from_training']]
        self.categorical_cols = [categorical_col['field_name'] for categorical_col in self.globals_context.cols_dict['categorical_cols'] if not categorical_col['exclude_feature_from_training']]
        self.numerical_cols = [numerical_col['field_name'] for numerical_col in self.globals_context.cols_dict['numerical_cols'] if not numerical_col['exclude_feature_from_training']]
        self.datetime_cols = [datetime_col['field_name'] for datetime_col in self.globals_context.cols_dict['datetime_cols'] if not datetime_col['exclude_feature_from_training']]
        self.selected_columns = self.boolean_cols + self.ordinal_cols + self.categorical_cols + self.numerical_cols + self.datetime_cols + [self.target_bucket_col_name]


    def feature_selection(self, dataset):
        return dataset[self.selected_columns]

    def basic_info(self, dataset):
        print(f"shape:\n {dataset.shape}")
        print(f"head:\n {dataset.head(10)}")
        print(f"columns:\n {dataset.columns}")
        print(f"unique:\n {dataset.nunique(axis=0)}")
        print(f"describe:\n {dataset.describe(include='all')}")

    def correlation(self, dataset):
        self.corr_boolean_cols = [boolean_col['field_name'] for boolean_col in self.globals_context.cols_dict['boolean_cols'] if boolean_col['include_in_correlation']]
        self.corr_ordinal_cols = [ordinal_col['field_name'] for ordinal_col in self.globals_context.cols_dict['ordinal_cols'] if ordinal_col['include_in_correlation']]
        self.corr_categorical_cols = [categorical_col['field_name'] for categorical_col in self.globals_context.cols_dict['categorical_cols'] if
                                 categorical_col['include_in_correlation']]
        self.corr_numerical_cols = [numerical_col['field_name'] for numerical_col in self.globals_context.cols_dict['numerical_cols'] if
                               numerical_col['include_in_correlation']]
        self.corr_datetime_cols = [datetime_col['field_name'] for datetime_col in self.globals_context.cols_dict['datetime_cols'] if datetime_col['include_in_correlation']]
        self.corr_columns = self.corr_boolean_cols + self.corr_ordinal_cols + self.corr_categorical_cols + self.corr_numerical_cols + self.corr_datetime_cols + [self.target_ordinal_bucket_col_name]
        dataset = self.globals_context.transform_delay_bucket_to_ordinal(dataset)
        dataset.pop('DelayBucket')
        dataset = dataset[self.corr_columns]
        corr = dataset.corr(numeric_only=True)
        corr = corr.round(3)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        df_mask = corr.mask(mask)
        fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(),
                                          x=df_mask.columns.tolist(),
                                          y=df_mask.columns.tolist(),
                                          colorscale=px.colors.diverging.RdBu,
                                          hoverinfo="none",  # Shows hoverinfo for null values
                                          showscale=True, ygap=1, xgap=1
                                          )
        fig.update_xaxes(side="bottom")

        fig.update_layout(
            title_text='Heatmap',
            title_x=0.5,
            width=1000,
            height=1000,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_zeroline=False,
            yaxis_zeroline=False,
            yaxis_autorange='reversed',
            template='plotly_white'
        )

        # NaN values are not handled automatically and are displayed in the figure
        # So we need to get rid of the text manually
        for i in range(len(fig.layout.annotations)):
            if fig.layout.annotations[i].text == 'nan':
                fig.layout.annotations[i].text = ""

        fig.show()

        # fig = go.Figure()
        # fig.add_trace(
        #     go.Heatmap(
        #         x=correlation.columns,
        #         y=correlation.index,
        #         z=np.array(correlation)
        #     )
        # )
        # fig.show()

    def plots_for_columns(self, x_column_name, y_column_name):
        self.scatter_plot(x_column_name, y_column_name)
        self.histogram(x_column_name, y_column_name)
        self.plotbox(x_column_name, y_column_name)

    def scatter_plot(self, x_column_name, y_column_name):
        fig = px.scatter(x=self.dataset[x_column_name], y=self.dataset[y_column_name])
        fig.show()

    def histogram(self, x_column_name, y_column_name):
        fig = px.histogram(self.dataset, x=x_column_name, y=y_column_name)
        fig.show()

    def plotbox(self, x_column_name, y_column_name):
        fig = px.box(self.dataset, x=x_column_name, y=y_column_name)
        fig.show()

    def scatter_matrix(self):
        fig = px.scatter_matrix(self.dataset)
        fig.show()
    def dataset_extract_target(self, dataset):
        target_col_name = 'ArrDelay'
        return self.globals_context.divide_target_to_buckets(self.dataset, target_col_name, self.target_bucket_col_name)

    def eda(self):
        dataset = self.dataset_extract_target(self.dataset)
        dataset = self.feature_selection(dataset)
        self.basic_info(dataset)
        self.correlation(dataset)
        self.plots_for_columns('CarrierDelay', self.target_bucket_col_name)
        self.plots_for_columns('WeatherDelay', self.target_bucket_col_name)
        self.plots_for_columns('NASDelay', self.target_bucket_col_name)
        self.plots_for_columns('SecurityDelay', self.target_bucket_col_name)
        self.plots_for_columns('LateAircraftDelay', self.target_bucket_col_name)


if __name__ == "__main__":
    fareMLPrediction = FareMLEDA()
    fareMLPrediction.eda()