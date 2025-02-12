from metaflow import FlowSpec, step, Parameter, IncludeFile


def script_path(filename):

    import os
    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


class MnistFlow(FlowSpec):
    """
        The flow performs the following steps:
        1) Ingest the MNIST csv data into Pandas DataFrame
        2) Clean and wrangle data
        3) Split data into train and test
        4) Fit model on train data (multiple models with branches)
        5) Predict on test data
        6) Evaluate result
    """

    mnist_train_data = IncludeFile('mnist_data',
        help='The path to mnist data file.',
        default=script_path('mnist_train.csv'))

    @step
    def start(self):
        """
            Load the data
        """
        import pandas as pd
        from io import StringIO

        # Read data from csv file
        self.mnist_df = pd.read_csv(StringIO(self.mnist_train_data))
        self.next(self.prepare_data)

    @step
    def prepare_data(self):

        """
            prepare data
        """
        # Extract the features and the label from the data
        self.X_df = self.mnist_df.drop(["label"], axis=1)
        self.Y_df = self.mnist_df.label.values

        self.next(self.split_data)

    @step
    def split_data(self):
        """
            Split train data for modelling
        """
        from sklearn.model_selection import train_test_split

        # Split data into train and test set
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_df, self.Y_df, test_size=0.5, random_state=42
            )
        self.next(self.fit_predict_model1, self.fit_predict_model2)

    @step
    def fit_predict_model1(self):
        """
            Fit a gaussian naive bayes model to the data
        """
        # Import model
        from sklearn.naive_bayes import GaussianNB

        modelA = GaussianNB()

        # Fit the model
        modelA.fit(self.X_train, self.Y_train)

        # Predict
        self.predictionA = modelA.predict(self.X_test)

        self.next(self.join)

    @step
    def fit_predict_model2(self):
        """
            Fit a Random forest model to the data
        """
        # Import model
        from sklearn.ensemble import RandomForestClassifier

        modelB = RandomForestClassifier(random_state=1)

        # Fit the model
        modelB.fit(self.X_train, self.Y_train)

        # Predict
        self.predictionB = modelB.predict(self.X_test)

        self.next(self.join)

    @step
    def join(self, inputs):
        """
            merge the data artifact from the models
        """

        # merge artificate during a join
        self.merge_artifacts(inputs)

        self.next(self.evaluate)

    @step
    def evaluate(self):
        """
            Evaluate the score of the models
        """

        from sklearn.metrics import accuracy_score

        # Measure accuracy
        print('Accuracy score for GaussianNB {}'.format(
            accuracy_score(self.predictionA, self.Y_test)
            ))
        print('Accuracy score for RandomForest {}'.format(
            accuracy_score(self.predictionB, self.Y_test)
            ))

        self.next(self.end)

    @step
    def end(self):
        """
            End of flow
        """

        print("finished")


if __name__ == '__main__':
    MnistFlow()
