from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class Reframer(BaseEstimator, TransformerMixin):
    def __init__(self, n_in=1, n_out=1, dropnan=True, columns=None):
        self.n_in = n_in
        self.n_out = n_out
        self.dropnan = dropnan
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_vars = 1 if type(X) is list else X.shape[1]
        dff = pd.DataFrame(X)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(self.n_in, 0, -1):
            cols.append(dff.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.n_out):
            cols.append(dff.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if self.dropnan:
            agg.dropna(inplace=True)
        return agg


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, n_in=1, n_out=1, n_vars=1, columns=None):
        self.n_in = n_in
        self.n_out = n_out
        self.columns = columns
        self.n_vars = n_vars

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #n_vars = 1 if type(X) is list else X.shape[1]
        #reframed_data = series_to_supervised(data, in_size, out_size)
        # drop columns we don't want to predict
        output_cols_start = self.n_in * self.n_vars
        cols_to_keep = [i for i in range(output_cols_start, len(X.columns), self.n_vars)]
        cols_to_drop = [i for i in range(output_cols_start, len(X.columns)) if i not in cols_to_keep]
        #drop_cols = [i for i in range(output_cols_start, len(X.columns), self.n_vars)]
        X_transformed = X.drop(X.columns[cols_to_drop], axis=1)

        return X_transformed


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
