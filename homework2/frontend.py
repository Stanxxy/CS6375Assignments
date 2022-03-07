import os
import model
import utils
import pandas as pd

DEFAULT_N_ITER = 1_000_000
DEFAULT_TOLERANCE = 1e-06
DEFAULT_LEARNING_RATE = 9e-05
DEFAULT_LOSS_GRAPH_PATH = "homemade_graph.png"
DEFAULT_LOG_PATH = "result.log"
DEFAULT_PANELTY = "l2"
DEFAULT_ALPHA = 1e-04


def load_data(path) -> pd.DataFrame:
    return utils.read_data(path)


def prepare_data(df: pd.DataFrame) -> tuple:
    df = utils.drop_columns(df)
    df = utils.convert_data_range(df)
    df = utils.convert_catagorical(df)
    training_set, testing_set = utils.train_test_split(df)
    scaled_training_X, training_y, scaled_testing_X, testing_y = utils.scale_dataset(
        training_set, testing_set)
    return {"training_x": scaled_training_X,
            "training_y": training_y,
            "testing_x": scaled_testing_X,
            "testing_y": testing_y}


def do_training(model_instance: model.Base_regression, data_dictionary: dict, param_dict: dict) -> model.Base_regression:
    res = utils.training_with_log(model_instance, data_dictionary, param_dict)
    with open(param_dict['log_path'], "a+") as f:
        f.write(res + "\n")
    # res
    return model_instance


def tuning():
    # change dictionary and do multiple training
    pass


# main function
if __name__ == '__main__':
    path = input("please input data path: ")
    if path.startswith("http://"):
        os.system("wget {}".format(path))
        table = load_data(path.split("/")[-1])
    elif path == "q":
        print("quit.")
        exit()
    else:
        table = load_data(path)
    data_dictionary = prepare_data(table)
    model_num = input("Which model do you wnat to run? (homemade/packaged)")
    if model_num == "homemade":
        # do training on homemade model
        model1 = model.Homemade_linear_regression()
        n_iter = input(
            "What is the top num of iterater? ({} by default) ".format(DEFAULT_N_ITER))
        tolerance = input(
            "What is the tolerance of loss change? ({} by default) ".format(DEFAULT_TOLERANCE))
        learning_rate = input(
            "What is the learning rate? ({} by default) ".format(DEFAULT_LEARNING_RATE))
        penalty = input(
            "What is the penalty tyoe? ({} by default) ".format(DEFAULT_PANELTY))
        alpha = input(
            "What is the weight do you want to set the for panelty? ({} by default) ".format(DEFAULT_ALPHA))
        training_graph_path = input(
            "Where do you want to save the graph of the loss curve? ({} by default) ".format(DEFAULT_LOSS_GRAPH_PATH))
        log_path = input(
            "Where do you want to save the logs? ({} by default) ".format(DEFAULT_LOG_PATH))
        param_dict = {
            # Items will be decided later
            "n_iter": int(n_iter) if n_iter != "" else DEFAULT_N_ITER,
            "tolerance": float(tolerance) if tolerance != "" else DEFAULT_TOLERANCE,
            "penalty": penalty if penalty != "" else DEFAULT_PANELTY,
            "alpha": float(alpha) if alpha != "" else DEFAULT_ALPHA,
            "learning_rate": float(learning_rate) if learning_rate != "" else DEFAULT_LEARNING_RATE,
            "loss_graph_path": training_graph_path if training_graph_path != "" else DEFAULT_LOSS_GRAPH_PATH,
            "log_path": log_path if log_path != "" else DEFAULT_LOG_PATH
        }
        do_training(model1, data_dictionary, param_dict)

    elif model_num == "packaged":
        # do training through packaged model
        model2 = model.Package_linear_regression()
        n_iter = input(
            "What is the max num of iterater? ({} by default) ".format(DEFAULT_N_ITER))
        tolerance = input(
            "What is the tolerance of loss change? ({} by default) ".format(DEFAULT_TOLERANCE))
        learning_rate = input(
            "What is the learning rate? ({} by default) ".format(DEFAULT_LEARNING_RATE))
        penalty = input(
            "What is the penalty tyoe? ({} by default) ".format(DEFAULT_PANELTY))
        alpha = input(
            "What is the weight do you want to set the for panelty? ({} by default) ".format(DEFAULT_ALPHA))
        log_path = input(
            "Where do you want to save the logs? ({} by default) ".format(DEFAULT_LOG_PATH))
        param_dict = {
            "max_iter": int(n_iter) if n_iter != "" else DEFAULT_N_ITER,
            "penalty": penalty if penalty != "" else DEFAULT_PANELTY,
            "learning_rate": float(learning_rate) if learning_rate != "" else DEFAULT_LEARNING_RATE,
            "verbose": 1,
            "tol": float(tolerance) if tolerance != "" else DEFAULT_TOLERANCE,
            "alpha": float(alpha) if alpha != "" else DEFAULT_ALPHA,
            "log_path": log_path if log_path != "" else DEFAULT_LOG_PATH
        }
        do_training(model2, data_dictionary, param_dict)
