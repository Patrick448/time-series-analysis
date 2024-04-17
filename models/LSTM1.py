from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline  # pipeline making

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM, Attention
from keras.layers import Dropout
from custom_transforms.transforms import *
from utils.utils import train_test_validation_split
from utils.utils import input_output_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class LSTM1:
    def __init__(self):
        self.rmse = None
        self.model = None
        self.rmse_by_timestep = None
        self.history = None

    def run(self, data, cols, in_size, out_size, keep_only_size):
        train, valid, test = train_test_validation_split(data, 0.7, 0.2)
        train_index, valid_index, test_index = train.index, valid.index, test.index

        # in_size = 8
        # out_size = 8
        # keep_only_size = 8

        input_columns = cols
        n_vars = len(input_columns)
        scaler = MinMaxScaler(feature_range=(0, 1))
        column_selector = ColumnSelector(input_columns)
        reframer = Reframer(n_in=in_size, n_out=out_size)
        drop_cols = DropColumns(n_in=in_size, n_out=out_size, n_vars=n_vars)

        preprocess_pipeline = Pipeline(
            [
                ('column_selector', column_selector),
                ('scaler', scaler),
                ('reframer', reframer),
                ('drop_cols', drop_cols)
            ]
        )

        preprocessed_train = preprocess_pipeline.fit_transform(train)
        preprocessed_valid = preprocess_pipeline.transform(valid)
        preprocessed_test = preprocess_pipeline.transform(test)

        train_X, train_Y = input_output_split(preprocessed_train, in_size, keep_only_size)
        validation_X, validation_Y = input_output_split(preprocessed_valid, in_size, keep_only_size)
        test_X, test_Y = input_output_split(preprocessed_test, in_size, keep_only_size)

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        validation_X = validation_X.reshape((validation_X.shape[0], 1, validation_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

        model = Sequential()
        model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
        #model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Dense(keep_only_size))
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model

        # fit network
        history = model.fit(train_X, train_Y, epochs=100, batch_size=70, validation_data=(validation_X, validation_Y),
                            verbose=2, shuffle=False)
        self.history = history.history
        # make a prediction
        yhat = model.predict(test_X)
        # test_X = test_X.reshape((test_X.shape[0], 16))
        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_Y, yhat))
        self.rmse = rmse
        print('Test RMSE: %.3f' % rmse)

        normalized_test = preprocess_pipeline[:-2].transform(test)
        test_price = normalized_test[:, 0]
        test_series = pd.Series(test_price, index=test_index)
        #test['Preco_unitario'].plot(label="actual_norm")
        rmses_list = []

        for i in range(keep_only_size):
            pred = yhat[:, i]  # , index=test_index[jump:len(yhat)+jump])
            jump = in_size + i - 1
            rmses_list.append(np.sqrt(mean_squared_error(test_series[jump:len(pred) + jump], pred)))

        self.rmse_by_timestep = pd.DataFrame(rmses_list, index=[i + 1 for i in range(keep_only_size)], columns=['RMSE'])
