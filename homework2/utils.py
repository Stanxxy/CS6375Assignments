import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import unittest
import os
import json as js
from model import Base_regression, Homemade_linear_regression, Package_linear_regression

# do data preprocessing
# data IO


def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# convert date
season_book = {
    "spring equinox": [3, 21],
    "autumn equinox": [9, 23],
    "summer solstice": [6, 22],
    "winter solstice": [12, 22]
}


def convert_year_to_season(year: float, month: float, day: float, season_book: dict) -> int:
    if (month < season_book["spring equinox"][0] and day < season_book["spring equinox"][1]) \
            or (month >= season_book["winter solstice"][0] and day >= season_book["winter solstice"][1]):
        return 0  # winter
    elif (month >= season_book["spring equinox"][0] and day >= season_book["spring equinox"][1]) \
            or (month < season_book["summer solstice"][0] and day < season_book["summer solstice"][1]):
        return 1  # spring
    elif (month >= season_book["summer solstice"][0] and day >= season_book["summer solstice"][1]) \
            or (month < season_book["autumn equinox"][0] and day < season_book["autumn equinox"][1]):
        return 2  # summer
    else:
        return 3  # autumn


def convert_date(df: pd.DataFrame) -> pd.DataFrame:
    df['season'] = df[['year', 'month', 'day']].apply(lambda x: convert_year_to_season(
        x['year'], x['month'], x['day'], season_book), axis=1)
    return df


def hours_split(hour: float) -> int:
    actual_hour = int(hour)
    if actual_hour > 2 and actual_hour <= 6:
        return 0
    elif actual_hour > 6 and actual_hour <= 10:
        return 1
    elif actual_hour > 10 and actual_hour <= 14:
        return 2
    elif actual_hour > 14 and actual_hour <= 18:
        return 3
    elif actual_hour > 18 and actual_hour <= 22:
        return 4
    else:
        return 5


def convert_catagorical(df: pd.DataFrame) -> pd.DataFrame:
    df['Iws_delta'] = df['Iws'].diff()
    df['Iws_delta'][df.index == 24] = 1.79
    df['Ir_p'] = df['Ir'].map(lambda x: int(x > 0))
    df['Is_p'] = df['Is'].map(lambda x: int(x > 0))
    df['hour_range'] = df['hour'].map(hours_split)
    df = convert_date(df)
    df_processed = df.drop(
        ['year', 'month', 'day', 'hour', 'Iws', 'Is', 'Ir', 'pm2.5'], axis=1)
    cbwd_oh = pd.get_dummies(df_processed['cbwd'], prefix="cbwd")
    season_oh = pd.get_dummies(df_processed['season'], prefix="season")
    hour_range_oh = pd.get_dummies(
        df_processed['hour_range'], prefix="hour_range")
    df_oh_new = pd.concat(
        [df_processed, cbwd_oh, season_oh, hour_range_oh], axis=1)
    df_oh = df_oh_new.drop(['cbwd', 'season', 'hour_range'], axis=1)
    return df_oh


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop('No', axis=1)
    df = df.dropna()
    return df


def convert_data_range(df: pd.DataFrame) -> pd.DataFrame:
    df['log pm2.5'] = df['pm2.5'].map(lambda x: np.log(x) if x > 0 else x)
    return df


def train_test_split(df: pd.DataFrame) -> pd.DataFrame:
    training_portion = 0.9
    testing_set = 0.1
    samples = df.__len__()
    training_set = df[:int(samples * training_portion)]
    testing_set = df[int(samples * training_portion):]
    return training_set, testing_set


def scale_dataset(training_set: pd.DataFrame, testing_set: pd.DataFrame) -> tuple:
    scaler = MinMaxScaler()
    scaler.fit(training_set[['Iws_delta', 'DEWP', 'TEMP', 'PRES']])
    scaled_training_set = scaler.transform(
        training_set[['Iws_delta', 'DEWP', 'TEMP', 'PRES']])
    scaled_testing_set = scaler.transform(
        testing_set[['Iws_delta', 'DEWP', 'TEMP', 'PRES']])
    training_y = training_set['log pm2.5'].values
    testing_y = testing_set['log pm2.5'].values
    training_X = training_set.drop(
        ['log pm2.5', 'Iws_delta', 'DEWP', 'TEMP', 'PRES'], axis=1).values
    testing_X = testing_set.drop(
        ['log pm2.5', 'Iws_delta', 'DEWP', 'TEMP', 'PRES'], axis=1).values
    scaled_training_X = np.concatenate(
        [training_X.data, scaled_training_set], axis=1)
    scaled_testing_X = np.concatenate(
        [testing_X.data, scaled_testing_set], axis=1)
    return scaled_training_X, training_y, scaled_testing_X, testing_y

# do pic plot


def plot_and_save(traing_curves: list, test_curves: list, path: str) -> None:
    epochs = range(1, len(traing_curves) + 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, traing_curves, 'bo', label='Training loss')
    # b is for "solid blue line"
    if test_curves is not None:
        plt.plot(epochs, test_curves, 'ro', label='Testing loss')
    plt.title('Training and testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.show()

# do history logging


def training_with_log(model_instance: Base_regression, data_dict: dict, param_dict: dict) -> str:
    # convert log to a string

    res = {}
    if isinstance(model_instance, Homemade_linear_regression):
        alpha = param_dict.get("alpha", 1e-04)
        n_iter = param_dict.get("n_iter", 1000)
        tolerance = param_dict.get("tolerance", 1e-06)
        learning_rate = param_dict.get("learning_rate", 9e-05)
        penalty = param_dict.get("penalty", "l2")
        model_instance.train(data_dict['training_x'],
                             data_dict['training_y'],
                             alpha=alpha,
                             penalty=penalty, n_iter=n_iter, tolerance=tolerance, learning_rate=learning_rate,
                             test_X=data_dict.get('testing_x', None), test_y=data_dict.get('testing_y', None))
        plot_and_save(model_instance.mse_training,
                      model_instance.mse_testing, param_dict['loss_graph_path'])
    else:
        max_iter = param_dict.get("max_iter", 1000)
        penalty = param_dict.get("penalty", "l2")
        alpha = param_dict.get("alpha", 1e-04)
        learning_rate = param_dict.get("learning_rate", 9e-05)
        verbose = param_dict.get("verbose", 1)
        tol = param_dict.get("tol", 1e-06)
        model_instance = Package_linear_regression(
            max_iter=max_iter, eta0=learning_rate, verbose=verbose, tol=tol, penalty=penalty, alpha=alpha)
        model_instance.train(data_dict['training_x'],
                             data_dict['training_y'])

    testing_pred_y = model_instance.predict(data_dict['testing_x'])
    mse = mean_squared_error(testing_pred_y, data_dict['testing_y'])
    res.update(param_dict)
    res['mse'] = mse
    return js.dumps(res)


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        path = input("please input data path: ")
        if path.startswith("http://"):
            os.system("wget {}".format(path))
            self.table = read_data(path.split("/")[-1])
        else:
            self.table = read_data(path)

    # def test_1_drop_columns(self):
        self.df = drop_columns(self.table)

    # def test_2_convert_data_range(self):
        self.df = convert_data_range(self.df)
        # print(self.df.head())

    # def test_3_convert_categorical(self):
        self.df = convert_catagorical(self.df)
        # print(self.df.head())

    # def test_4_train_test_split(self):
        self.training_set, self.testing_set = train_test_split(self.df)

    # def test_5_scale_dataset(self):
        scaled_training_X, training_y, scaled_testing_X, testing_y = scale_dataset(
            self.training_set, self.testing_set)
        self.data_dict = {"training_x": scaled_training_X,
                          "training_y": training_y,
                          "testing_x": scaled_testing_X,
                          "testing_y": testing_y}
        # return super().setUp()

    def test_6_training_with_log(self):
        model1 = Homemade_linear_regression()
        model2 = Package_linear_regression()
        param_dict = {"loss_graph_path": "UT_graph.png"}
        js_str = training_with_log(model1, self.data_dict, param_dict)
        print(js_str)
        js_str2 = training_with_log(model2, self.data_dict, param_dict)
        print(js_str2)


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestLogin('test_login_blog'))
    # suite.addTest(TestLogin('test_add_essay'))
    # suite.addTest(TestLogin('test_release_essay'))
    # suite.addTest(TestLogin('test_quit_blog'))

    # runner = unittest.TextTestRunner()
    # runner.run(suite)
