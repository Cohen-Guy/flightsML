import os
import time
import pandas as pd

class GlobalsContextClass:

    def __init__(self, timestr):
        self.timestr = timestr
        self.debug_flag = True
        self.set_column_definitions()

    def set_column_definitions(self):
        self.cols_dict = {
            'boolean_cols':
                [
                    {
                        'field_name': 'Diverted',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'DivReachedDest',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'Cancelled',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                ],
            'ordinal_cols':
                [
                    {
                        'field_name': 'Flights',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'CarrierDelay',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'WeatherDelay',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'NASDelay',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'SecurityDelay',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'LateAircraftDelay',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'DivAirportLandings',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                ],
            'categorical_cols':
                [
                    # excluded: "Reporting_Airline", "DOT_ID_Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline", "OriginAirportSeqID", "OriginCityMarketID", "Origin", "OriginCityName", "OriginState", "OriginStateFips",
                    #           "OriginStateName", "OriginWac" "DestAirportSeqID", "DestCityMarketID", "Dest", "DestCityName", "DestState", "DestStateFips", "DestStateName", "DestWac",
                    #           "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15", "DepartureDelayGroups", "DepTimeBlk", "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn", "CRSArrTime", "ArrTime",
                    #           "ArrDelayMinutes", "ArrDel15", "ArrivalDelayGroups", "ArrTimeBlk", "DistanceGroup", "FirstDepTime", "Div1Airport", "Div1AirportID", "Div1AirportSeqID",
                    #          "Div1WheelsOn", "Div1WheelsOff", "Div1TailNum", "Div2Airport", "Div2AirportID", "Div2AirportSeqID", "Div2WheelsOn", "Div2TotalGTime", "Div2LongestGTime", "Div2WheelsOff",
                    #          "Div2TailNum", "Div3Airport","Div3AirportID", "Div3AirportSeqID", "Div3WheelsOn", "Div3TotalGTime", "Div3LongestGTime", "Div3WheelsOff", "Div3TailNum", "Div4Airport",
                    #          "Div4AirportID", "Div4AirportSeqID", "Div4WheelsOn", "Div4TotalGTime", "Div4LongestGTime", "Div4WheelsOff", "Div4TailNum", "Div5Airport", "Div5AirportID", "Div5AirportSeqID",
                    #          "Div5WheelsOn", "Div5TotalGTime", "Div5LongestGTime", "Div5WheelsOff", "Div5TailNum"
                    {
                        'field_name': 'Year',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'Quarter',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'Month',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'DayofMonth',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'DayOfWeek',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'IATA_CODE_Reporting_Airline',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'OriginAirportID',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'DestAirportID',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'CancellationCode',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'OriginCityName',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'DestCityName',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                ],
            'numerical_cols':
                [
                    {
                        'field_name': 'ArrDelay',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'AirTime',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'Distance',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'TotalAddGTime',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'LongestAddGTime',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'DivArrDelay',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'DivDistance',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'Div1TotalGTime',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'Div1LongestGTime',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    },
                    {
                        'field_name': 'ActualElapsedTime',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': False,
                    }
                ],
            'datetime_cols':
                [{
                    'field_name': 'CRSElapsedTime',
                    'description': '',
                    'exclude_feature_from_training': False,
                    'include_in_correlation': False,
                }],
            'special_handling_cols':
                [],
            'target_col':
                {
                    'field_name': 'DelayBucket',
                    'description': '',
                    'exclude_feature_from_training': False,
                    'include_in_correlation': True,
                },
        }

    def transform_delay_bucket_to_ordinal(self, dataset):
        dataset['DelayBucketOrdinal'] = dataset['DelayBucket']
        mapping = {'{-10000, -60}': -5, '{-60, -45}': -4, '{-45, -30}': -3, '{-30, -15}': -2, '{-15, -3}':-1, '{-3, 3}':0, '{3, 15}': 1, '{15, 30}':2, '{30, 45}':3, '{45, 60}':4, '{60, 10000}': 5}
        dataset = dataset.replace({'DelayBucketOrdinal': mapping})
        dataset['DelayBucketOrdinal'] = dataset['DelayBucketOrdinal'].astype('Int64')
        return dataset

    def divide_target_to_buckets(self, dataset, target_col_name, target_bucket_col_name):
        dataset.loc[dataset[target_col_name].between(-10000, -60, 'both'), target_bucket_col_name] = '{-10000, -60}'
        dataset.loc[dataset[target_col_name].between(-60, -45, 'right'), target_bucket_col_name] = '{-60, -45}'
        dataset.loc[dataset[target_col_name].between(-45, -30, 'right'), target_bucket_col_name] = '{-45, -30}'
        dataset.loc[dataset[target_col_name].between(-30, -15, 'right'), target_bucket_col_name] = '{-30, -15}'
        dataset.loc[dataset[target_col_name].between(-15, -3, 'right'), target_bucket_col_name] = '{-15, -3}'
        dataset.loc[dataset[target_col_name].between(-3, 3, 'right'), target_bucket_col_name] = '{-3, 3}'
        dataset.loc[dataset[target_col_name].between(3, 15, 'right'), target_bucket_col_name] = '{3, 15}'
        dataset.loc[dataset[target_col_name].between(15, 30, 'right'), target_bucket_col_name] = '{15, 30}'
        dataset.loc[dataset[target_col_name].between(30, 45, 'right'), target_bucket_col_name] = '{30, 45}'
        dataset.loc[dataset[target_col_name].between(45, 60, 'right'), target_bucket_col_name] = '{45, 60}'
        dataset.loc[dataset[target_col_name].between(60, 10000, 'right'), target_bucket_col_name] = '{60, 10000}'
        return dataset

    def feature_engineering(self, X):
        X['Year'] = pd.to_datetime(X['Year'], format='%Y')
        X['Month'] = pd.to_datetime(X['Month'], format='%m')
        X['FlightDate'] = pd.to_datetime(X['FlightDate'], format='%Y-%m-%d')
        return X