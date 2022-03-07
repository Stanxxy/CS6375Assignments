# CS6375 Assignment 2

This is the readme file for CS6375.502. This project is for the first assignment of Machine Learning course in Spring 22.

## How to set up environment

- In order to run the code in this project on UTD server, you should first set up the environment with pip:

```
python3 -m venv $PATH_TO_NEW_VIRTUAL_ENVIRONMENT
$PYTHON_ENVIRMENT/bin/pip install --target=$(pwd) requirements.txt
```

- This would help you to install all the necessary packages in the current directory. You could also ignore the "--target=" option and install packages directly if you have the sudo previledge. If you ignore "--target", you would install those packages globally

## How to run the code

- Use the following method to run the code:

```
$PYTHON_ENVIRMENT/bin/python NeuralNet-2.py $YOUR_DATASET_PATH
```

- If you use the default python on UTD server, remember to replace "$PYTHON_ENVIRMENT/bin/python" with "python3". All the code must be executed with python 3.

## How to interact with NeuralNet.py

- after training, python would automatically save the plots. No further interactions are required

* The program would put results with the same activation functions and max iteration numbers on the same plot.

## About project architecture

- Code for model and main function is in NeuralNet-2.py.
  - MLPRegressor is used to do regressor.
- Code for preprocessing dataset and plot results locates in utils.py

## Dataset hosted

- The dataset is hosted by UTD webservice at: https://personal.utdallas.edu/~sxl200012/PRSA_data_2010.1.1-2014.12.31.csv

* The original data is from https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
