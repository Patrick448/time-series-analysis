import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline  # pipeline making

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential, Model
from keras.layers import LSTM, Attention, GRU, Flatten, Input, Permute, Concatenate
from keras.layers import Dropout

from custom_transforms.transforms import *
from utils.utils import train_test_validation_split
from utils.utils import input_output_split
from sklearn.metrics import (mean_squared_error,
                             r2_score, mean_absolute_error,
                             mean_absolute_percentage_error)
from keras.callbacks import EarlyStopping
import numpy as np


class LSTM1:
    def __init__(self):
        self.rmse = None
        self.mae = None
        self.mape = None
        self.mse = None
        self.r2 = None
        self.model = None
        self.rmse_by_timestep = None
        self.mae_by_timestep = None
        self.mape_by_timestep = None
        self.mse_by_timestep = None
        self.r2_by_timestep = None
        self.history = None

    def save_pred_ref(self, ref_pred_path, pred, ref, model_id):
        try:
            os.mkdir(ref_pred_path)
        except:
            pass
        pd.DataFrame(pred).to_csv(f'{ref_pred_path}/pred_{model_id}.csv')
        pd.DataFrame(ref).to_csv(f'{ref_pred_path}/ref_{model_id}.csv')

    def _create_simple_lstm(self, input_shape: tuple, output_shape: int) -> Model:
        input = Input(input_shape)
        lstm = LSTM(200, return_sequences=True, activation="tanh")(input)
        dropout = Dropout(0.2)(lstm)
        flatten = Flatten()(dropout)
        dense1 = Dense(10*output_shape)(flatten)
        output = Dense(output_shape)(dense1)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def _create_dia_lstm(self, input_shape: tuple, output_shape: int) -> Model:
        input = Input(input_shape)
        permuted_input = Permute((2, 1))(input)
        temporal_attention = Attention()(permuted_input)
        permute_temporal_attention = Permute((2, 1))(temporal_attention)
        feature_attention = Attention()(input)
        concatenate = Concatenate([permute_temporal_attention, feature_attention])
        lstm = LSTM(200, return_sequences=True, activation="tanh")(concatenate)
        dropout = Dropout(0.2)(lstm)
        flatten = Flatten()(dropout)
        dense1 = Dense(10*output_shape)(flatten)
        output = Dense(output_shape)(dense1)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='mean_squared_error', optimizer='adam')

        print(model.summary())
        return model


    def run(self, data, cols, in_size, out_size, keep_only, architecture, save_path=None, model_id=None, start_offset=None, end_offset=None):
        train, valid, test = train_test_validation_split(data, 0.7, 0.2)
        train_index, valid_index, test_index = train.index, valid.index, test.index
        keep_only_size = 1 if keep_only is not None else out_size
        input_columns = cols
        n_vars = len(input_columns)
        scaler = MinMaxScaler(feature_range=(0, 1))
        column_selector = ColumnSelector(input_columns)
        reframer = Reframer(n_in=in_size, n_out=out_size)
        drop_cols = DropColumns(n_in=in_size, n_out=out_size, n_vars=n_vars, keep_only=keep_only)

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

        if start_offset:
            preprocessed_test = preprocessed_test[(start_offset-1):]
        if end_offset:
            preprocessed_test = preprocessed_test[:-end_offset]


        train_X, train_Y = input_output_split(preprocessed_train, in_size, keep_only_size)
        validation_X, validation_Y = input_output_split(preprocessed_valid, in_size, keep_only_size)
        test_X, test_Y = input_output_split(preprocessed_test, in_size, keep_only_size)

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        validation_X = validation_X.reshape((validation_X.shape[0], 1, validation_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)



        #model = Sequential()
       # model.add(LSTM(6, return_sequences=True, activation="tanh", input_shape=(train_X.shape[1], train_X.shape[2])))
       # model.add(Dropout(0.2))
       # model.add(Flatten())
       # model.add(Dense(10))
       # model.add(Dense(keep_only_size))
       # model.compile(loss='mean_squared_error', optimizer='adam')

        if architecture == 'simple_lstm_v0':
            model = self._create_simple_lstm((train_X.shape[1], train_X.shape[2]), keep_only_size)
        elif architecture == 'dia_lstm_v0':
            model = self._create_dia_lstm((train_X.shape[1], train_X.shape[2]), keep_only_size)
        else:
            raise ValueError(f'Architecture {architecture} not found')

        model_checkpoint_callback = None
        if save_path:
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=save_path+".keras",
                monitor='val_loss',
                mode='min',
                save_best_only=True)

        # fit network
        history = model.fit(train_X, train_Y, epochs=100, batch_size=200,
                            validation_data=(validation_X, validation_Y),
                            verbose=2, shuffle=False,# use_multiprocessing=True,
                            callbacks=[EarlyStopping(patience=10, monitor='val_loss'),
                                       model_checkpoint_callback])
        self.history = history.history
        # make a prediction

        self.model = keras.models.load_model(save_path+".keras")
        yhat = self.model.predict(test_X)
        yhat = yhat.reshape((yhat.shape[0], keep_only_size))
       # test_X = test_X.reshape((test_X.shape[0], 16))
        # calculate RMSE

        # ----------------- DENORMALIZE

        denorm_test_Y = np.copy(test_Y)
        denorm_yhat = np.copy(yhat)

        for i, col in enumerate(denorm_test_Y.T):
            denorm_test_Y[:, i] = denormalize_with(col, len(cols), scaler, 0)

        for i, col in enumerate(denorm_yhat.T):
            denorm_yhat[:, i] = denormalize_with(col, len(cols), scaler, 0)

        test_Y = denorm_test_Y
        yhat = denorm_yhat

        # -----------------

        self.save_pred_ref("pred_ref", yhat, test_Y, model_id)

        rmse = np.sqrt(mean_squared_error(test_Y, yhat))
        mae = mean_absolute_error(test_Y, yhat)
        mape = mean_absolute_percentage_error(test_Y, yhat)
        mse = mean_squared_error(test_Y, yhat)
        r2 = r2_score(test_Y, yhat)

        self.mse = mse
        self.mape = mape
        self.rmse = rmse
        self.mae = mae
        self.r2 = r2


        rmses_list = []
        mae_list = []
        mse_list = []
        mape_list = []
        r2_list = []

        # todo: verificar se isso está correto
        for i in range(keep_only_size):
            pred = yhat[:, i]
            ref = test_Y[:, i]
            rmses_list.append(np.sqrt(mean_squared_error(ref, pred)))
            mae_list.append(mean_absolute_error(ref, pred))
            mse_list.append(mean_squared_error(ref, pred))
            mape_list.append(mean_absolute_percentage_error(ref, pred))
            r2_list.append(r2_score(ref, pred))

        self.rmse_by_timestep = pd.DataFrame(rmses_list, index=[i + 1 for i in range(keep_only_size)], columns=['RMSE'])
        self.mae_by_timestep = pd.DataFrame(mae_list, index=[i + 1 for i in range(keep_only_size)], columns=['MAE'])
        self.mse_by_timestep = pd.DataFrame(mse_list, index=[i + 1 for i in range(keep_only_size)], columns=['MSE'])
        self.mape_by_timestep = pd.DataFrame(mape_list, index=[i + 1 for i in range(keep_only_size)], columns=['MAPE'])
        self.r2_by_timestep = pd.DataFrame(r2_list, index=[i + 1 for i in range(keep_only_size)], columns=['R2'])

