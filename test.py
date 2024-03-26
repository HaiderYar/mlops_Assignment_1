import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class TestLinearRegressionModel(unittest.TestCase):
    def setUp(self):
        # Load the dataset
        self.data = pd.read_csv('SongDetails.csv')

    def test_load_dataset(self):
        # Check if the dataset is loaded properly
        self.assertFalse(self.data.empty, "Dataset is empty")

    def test_split_data(self):
        # Split the data into features (X) and target variable (y)
        X = self.data.drop(['popularity'], axis=1)
        y = self.data['popularity']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Check if the data is split correctly
        self.assertEqual(len(X_train), int(0.8 * len(X)), "Incorrect train-test split size")

    def test_model_performance(self):
        # Load the dataset
        data = pd.read_csv('SongDetails.csv')

        # Split the data into features (X) and target variable (y)
        X = data.drop(['popularity'], axis=1)
        y = data['popularity']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the Linear Regression model
        model = LinearRegression()

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Predict on the testing data
        y_pred = model.predict(X_test)

        # Evaluate the model using Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)

        # Check if Mean Squared Error is within acceptable range
        self.assertLessEqual(mse, 500, "Mean Squared Error too high")

if __name__ == '__main__':
    unittest.main()
