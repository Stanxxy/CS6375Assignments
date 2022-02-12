# CS6375 Assignment 1

This is the readme file for CS6375.502. This project is for the first assignment of Machine Learning course in Spring 22.

## How to set up environment

- In order to run the code in this project on UTD server, you should first set up the environment with pip:

```
pip install --target=$(pwd) requirements.txt
```

- This would help you to install all the necessary packages in the current directory. You could also ignore the "--target=" option and install packages directly if you have the sudo previledge. If you ignore "--target", you would install those packages globally

## How to run the code

- Use the following method to run the code:

```
$PYTHON_ENVIRMENT/bin/python frontend.py
```

- If you use the default python on UTD server, remember to replace "$PYTHON_ENVIRMENT/bin/python" with "python3". All the code must be executed with python 3.

## How to interact with frontend

- Follow the instruction to interact with frontend. As an MVP, I haven't deal with the robustness issue. Please follow the instructions and input values in the right type. Otherwise there will be execeptions.

## About project architecture

- Code for model is in model.py, including two model classes derived from a base class Base_regression.
  - Train and predict method are provided as the basic functions.
  - For the homemade regressor, learning rate, iterate number upper bound and tolerace are adjustable. There are the three parameters that are tuned.
  - For the frontend regressor, learning rate, tolerance, iterate number upperbound, and learning rate type are tunable.
  - history are recorded in the log_path
- Code for preprocessing and other functional methods locates in utils.py
- Code for intraction with users are put in the frontend.py
