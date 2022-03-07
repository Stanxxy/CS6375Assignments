#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################

from sklearn.neural_network import MLPRegressor
import utils
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)

    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc

    def preprocess(self):
        self.processed_data = self.raw_input
        self.df = utils.drop_columns(self.processed_data)
        self.df = utils.convert_data_range(self.df)

        self.df = utils.convert_catagorical(self.df)
        self.training_set, self.testing_set = train_test_split(self.df)

        self.scaled_training_X, self.training_y, self.scaled_testing_X, self.testing_y = utils.scale_dataset(
            self.training_set, self.testing_set)
        # return super().setUp()

    # def test_6_training_with_log(self):
        # model1 = Homemade_linear_regression()
        # model2 = Package_linear_regression()
        # param_dict = {"loss_graph_path": "UT_graph.png"}
        # js_str = training_with_log(model1, self.data_dict, param_dict)
        # print(js_str)
        # js_str2 = training_with_log(model2, self.data_dict, param_dict)
        # print(js_str2)
        # return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        # ncols = len(self.processed_data.columns)
        # nrows = len(self.processed_data.index)
        # X = self.processed_data.iloc[:, 0:(ncols - 1)]
        # y = self.processed_data.iloc[:, (ncols-1)]
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y)

        self.data_dict = {"training_x": self.scaled_training_X,
                          "training_y": self.training_y,
                          "testing_x": self.scaled_testing_X,
                          "testing_y": self.testing_y}

        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200]  # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics
        history_dict = {}
        for ac in activations:
            history_dict[ac] = {}
            for mx_it in max_iterations:
                history_dict[ac][mx_it] = {}
                for num_hidden_layer in num_hidden_layers:
                    for lr in learning_rate:
                        model = MLPRegressor(hidden_layer_sizes=[num_hidden_layer, 1],
                                             activation=ac, learning_rate_init=lr, verbose=False)
                        training_mse_history = [None] * mx_it
                        testing_mse_history = [None] * mx_it
                        for epoch in range(mx_it):
                            model.partial_fit(
                                self.data_dict["training_x"], self.data_dict["training_y"])
                            pred_training_y = model.predict(
                                self.data_dict["training_x"])
                            pred_testing_y = model.predict(
                                self.data_dict["testing_x"])
                            mse_train = mean_squared_error(
                                pred_training_y, self.data_dict["training_y"])
                            mse_test = mean_squared_error(
                                pred_testing_y, self.data_dict["testing_y"])
                            training_mse_history[epoch] = mse_train
                            testing_mse_history[epoch] = mse_test
                        history_dict[ac][mx_it][(num_hidden_layer, lr)] = (
                            training_mse_history, testing_mse_history)

                    # res.update(param_dict)
                    # res['mse'] = mse
                    # js.dumps(res)
                    # loss_graph_path = "hidden_layers_{}-learning_rate_{}-activation_{}-max_iter_{}".\
                    #     format(num_hidden_layers, lr, ac, mx_it)

        utils.plot_and_save(history_dict)  # training_mse_history,
        # testing_mse_history, loss_graph_path)
        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0


if __name__ == "__main__":
    # put in path to your file
    dataset_path = sys.argv[1]
    neural_network = NeuralNet(dataset_path)
    neural_network.preprocess()
    neural_network.train_evaluate()
