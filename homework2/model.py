import unittest
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import datetime
# models for question 1 and 2


class Base_regression:
    def __init__(self) -> None:
        pass

    def train(self) -> None:
        pass

    def predict(self) -> np.array:
        pass


class Homemade_linear_regression(Base_regression):
    def __init__(self, intercept=True) -> None:
        self.intercept = intercept
        self.training_history = {}

    def train(self, X: np.array, y: np.array, penalty='l2', alpha=1e-04, n_iter=10000, tolerance=1e-06, learning_rate=9e-05, test_X=None, test_y=None) -> None:
      # X is 2d array, y is 1d array
        n = X.shape[0]
        if test_X is not None:
            n_test = test_X.shape[0]
        d = X.shape[1] + 1
        self.weight = np.random.randn(d).reshape(-1, 1)
        if self.intercept:
            X = np.concatenate([X, np.ones((n, 1))], axis=1)
            if test_X is not None:
                test_X = np.concatenate([test_X, np.ones((n_test, 1))], axis=1)
        self.weight = self.gradient_descent(
            X, y, self.weight, n_iter, tolerance, learning_rate, test_X, test_y, alpha, penalty)

    def predict(self, X: np.array) -> np.array:
        n = X.shape[0]
        if self.intercept:
            X = np.concatenate([X, np.ones((n, 1))], axis=1)
        return np.dot(X, self.weight).reshape(-1)

    def gradient_descent(self, X: np.array, y: np.array, start, n_iter, tolerance,
                         learning_rate, test_X, test_y, alpha: float, penalty: str) -> np.array:
        mse_training = []
        mse_testing = []
        vector = start
        last_loss = 0
        start_time = datetime.datetime.now()
        for i in range(n_iter):
            diff = - learning_rate * \
                self.gradient(X, y, vector, alpha, penalty)
            # if np.all(np.abs(diff) <= tolerance):
            # break
            # mean square err on training set and test set
            mse_training_i = mean_squared_error(
                self.predict(X[:, :-1]), y)
            # print(np.abs(mse_training - last_loss))
            mse_training.append(mse_training_i)

            vector += diff
            self.weight = vector

            if test_X is not None and test_y is not None:
                mse_testing_i = mean_squared_error(
                    self.predict(test_X[:, :-1]), test_y)
                mse_testing.append(mse_testing_i)

            if np.abs(mse_training_i - last_loss) <= tolerance:
                break

            if i % 100 == 0:
                end_time = datetime.datetime.now()
                print("--epoch {}".format(i))
                w = self.weight.reshape(-1)
                # print(w)
                # print(mse_training)
                print("Norm: {:.7f}, NNZs: {}, Bias: {:.7f}, T: {}, Avg. loss: {:.7f}".format(
                    np.linalg.norm(w[:-1]), w.shape[0] - 1, w[-1], i*X.shape[0], mse_training_i))
                print("Total training time: {:.2f} seconds.".format(
                    (end_time - start_time).total_seconds()))
                start_time = datetime.datetime.now()
                # learning_rate /= 2
            last_loss = mse_training_i

        self.mse_training = mse_training
        if test_X is not None and test_y is not None:
            self.mse_testing = mse_testing
        else:
            self.mse_testing = None
        return vector

    def gradient(self, X: np.array, y: np.array, weight: np.array, alpha, penalty) -> float:
        n = X.shape[0]
        residual = (np.dot(X, weight) - y.reshape(-1, 1))

        res = np.dot(X.transpose(), residual) / n
        if penalty == "l2":
            res += alpha * np.sqrt(np.linalg.norm(weight)) / n
        return res

# question2


class Package_linear_regression(Base_regression):
    def __init__(self, intercept=True, max_iter=10000, learning_rate="constant", eta0=9e-05, verbose=1, tol=1e-06, alpha=1e-04, penalty="l2") -> None:
        SGDRegressor()
        self.model = SGDRegressor(
            max_iter=max_iter, learning_rate=learning_rate, eta0=eta0, verbose=verbose, tol=tol, fit_intercept=intercept, alpha=alpha, penalty=penalty)
        self.intercept = intercept

    def train(self, X: np.array, y: np.array) -> None:
      # X is 2d array, y is 1d array
        # n = X.shape[0]
        d = X.shape[1] + 1
        self.weight = np.zeros(d)
        # if self.intercept:
        #     X = np.concatenate([X, np.ones((n, 1))], axis=1)
        self.model.fit(X, y)
        self.weight = np.array([self.model.coef_, self.model.intercept_])

    def predict(self, X) -> np.array:
        # if self.intercept:
        #     n = X.shape[0]
        # X = np.concatenate([X, np.ones((n, 1))], axis=1)
        return self.model.predict(X)


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        x = np.array(list(range(100))).reshape(-1, 1)
        y_raw = 5 * x.reshape(-1) + 5
        # print(x)
        y_noise = np.random.randn(y_raw.__len__())
        y = y_raw + y_noise
        # print(y)
        self.model_home_made = Homemade_linear_regression()
        self.model_packaged = Package_linear_regression()
        self.training_x, self.training_y, self.testing_x, self.testing_y = x[
            :80], y[:80], x[80:], y[80:]
        self.tol = 0.01
        return super().setUp()

    def test_homemade_reg(self):

        self.model_home_made.train(self.training_x,
                                   self.training_y, test_X=self.testing_x, test_y=self.testing_y)
        print(self.model_home_made.weight.reshape(-1))
        # self.assertTrue(
        #     np.abs(self.model_home_made.weight[0][0] - 3) < self.tol)
        # self.assertTrue(
        #     np.abs(self.model_home_made.weight[1][0] - 5) < self.tol)

    def test_package_reg(self):
        self.model_packaged.train(self.training_x,
                                  self.training_y)
        print(self.model_packaged.weight.reshape(-1))
        # self.assertTrue(np.abs(self.model_packaged.model.coef_[
        #                        0] - 3) < self.tol)
        # self.assertTrue(np.abs(self.model_packaged.model.coef_[
        #                        1] - 5) < self.tol)


if __name__ == '__main__':
    unittest.main()
