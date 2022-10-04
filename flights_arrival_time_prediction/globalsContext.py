import os
import time


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
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'DivReachedDest',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'Cancelled',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                ],
            'ordinal_cols':
                [
                    {
                        'field_name': 'Flights',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'CarrierDelay',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'WeatherDelay',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'NASDelay',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'SecurityDelay',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'LateAircraftDelay',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'DivAirportLandings',
                        'description': '',
                        'exclude_feature_from_training': False,
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
                    },
                    {
                        'field_name': 'Quarter',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'Month',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'DayofMonth',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'DayOfWeek',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'IATA_CODE_Reporting_Airline',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'OriginAirportID',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'DestAirportID',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'CancellationCode',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'OriginCityName',
                        'description': '',
                        'exclude_feature_from_training': True,
                    },
                    {
                        'field_name': 'DestCityName',
                        'description': '',
                        'exclude_feature_from_training': True,
                    },
                ],
            'numerical_cols':
                [
                    {
                        'field_name': 'ArrDelay',
                        'description': '',
                        'exclude_feature_from_training': True,
                    },
                    {
                        'field_name': 'AirTime',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'Distance',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'TotalAddGTime',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'LongestAddGTime',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'DivArrDelay',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'DivDistance',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'Div1TotalGTime',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                    {
                        'field_name': 'Div1LongestGTime',
                        'description': '',
                        'exclude_feature_from_training': False,
                    },
                ],
            'datetime_cols':
                [{
                    'field_name': 'CRSElapsedTime',
                    'description': '',
                    'exclude_feature_from_training': False,
                }],
            'special_handling_cols':
                [{
                    'field_name': 'ActualElapsedTime',
                    'description': '',
                    'exclude_feature_from_training': False,
                }],
            'target_col':
                {
                    'field_name': 'DelayBucket',
                    'description': '',
                    'exclude_feature_from_training': False,
                },
        }