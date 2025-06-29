import os

import pandas as pd
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline  # pipeline making

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential, Model
from keras.layers import LSTM, Attention, GRU, Flatten, Input, Permute, Concatenate
from keras.layers import Dropout
from statsmodels.tsa.statespace.sarimax import SARIMAX

from custom_transforms.transforms import *
from utils.utils import train_test_validation_split
from utils.utils import input_output_split
from sklearn.metrics import (mean_squared_error,
                             r2_score, mean_absolute_error,
                             mean_absolute_percentage_error)
from keras.callbacks import EarlyStopping
import numpy as np


class ModelTraining:
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

    def save_pred_ref(self, ref_pred_path, pred, ref,  model_id, dt=None, extra=None):
        try:
            os.mkdir(ref_pred_path)
        except:
            pass
        pred_df =pd.DataFrame(pred)
        ref_df =pd.DataFrame(ref)

        if dt is not None:
            pred_df.index = dt
            ref_df.index= dt

        if extra is not None:
            pred_df['extra'] = extra

        pred_df.to_csv(f'{ref_pred_path}/pred_{model_id}.csv')
        ref_df.to_csv(f'{ref_pred_path}/ref_{model_id}.csv')

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

    def run(self, data, cols, in_size, out_size, keep_only, architecture, save_path=None, model_id=None, start_offset=None, end_offset=None, train_valid_test: tuple = None):
        train, valid, test = train_test_validation_split(data, 0.7, 0.2, train_valid_test=train_valid_test)
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

        self.save_pred_ref("pred_ref", yhat, test_Y, model_id,test[in_size-1:-out_size].index)

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

    def preprocess_data(self, data, cols, in_size, out_size, keep_only, architecture, save_path=None, model_id=None, start_offset=None, end_offset=None, train_valid_test: tuple = None):
        keep_only_size = 1 if keep_only is not None else out_size
        input_columns = cols
        n_vars = len(input_columns)


        tscv_split = TimeSeriesSplit(test_size=in_size+out_size, n_splits=10)

        return tscv_split

    def get_LSTM_preprocess_pipeline(self, input_columns, in_size, out_size, keep_only) -> Pipeline:

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

        return preprocess_pipeline

    def lstm_train_predict(self, train, test, identifier, cols, in_size, out_size, keep_only, architecture, save_path=None):

        # daqui pra frente são coisas especificas do modelo
        # todo: talvez o scaler possa ser passado para a etapa anterior

        n_vars = len(cols)
        keep_only_size = 1 if keep_only is not None else out_size

        scaler = MinMaxScaler(feature_range=(0, 1))
        column_selector = ColumnSelector(cols)
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

        model_path = f"{save_path}.t{identifier}.keras"
        test = pd.concat([train.tail(in_size), test], axis="rows")

        train_pp = preprocess_pipeline.fit_transform(train)
        test_pp = preprocess_pipeline.transform(test)

        train_X, train_Y = input_output_split(train_pp, in_size, out_size)
        test_X, test_Y = input_output_split(test_pp, in_size, out_size)

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        if architecture == 'simple_lstm_v0':
            model = self._create_simple_lstm((train_X.shape[1], train_X.shape[2]), keep_only_size)
        elif architecture == 'dia_lstm_v0':
            model = self._create_dia_lstm((train_X.shape[1], train_X.shape[2]), keep_only_size)
        else:
            raise ValueError(f'Architecture {architecture} not found')

        model_checkpoint_callback = None
        if save_path:
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                mode='min',
                save_best_only=True)

        # fit network
        history = model.fit(train_X, train_Y, epochs=100, batch_size=200,
                            # validation_data=(validation_X, validation_Y),
                            verbose=2, shuffle=False,  # use_multiprocessing=True,
                            validation_split=0.1,
                            callbacks=[EarlyStopping(patience=10, monitor='val_loss'),
                                       model_checkpoint_callback])
        self.history = history.history
        # make a prediction

        self.model = keras.models.load_model(model_path)

        yhat = self.model.predict(test_X)
        denorm_test_Y = np.copy(test_Y)
        denorm_yhat = np.copy(yhat)

        for i, col in enumerate(denorm_test_Y.T):
            denorm_test_Y[:, i] = denormalize_with(col, len(cols), scaler, 0)

        for i, col in enumerate(denorm_yhat.T):
            denorm_yhat[:, i] = denormalize_with(col, len(cols), scaler, 0)

        #test_Y = denorm_test_Y
        #yhat = denorm_yhat

        test_Y_series_date = pd.Series(denorm_test_Y[0], index=test.index[-out_size:])
        yhat_series_date = pd.Series(denorm_yhat[0], index=test.index[-out_size:])

        return model_path, test_Y_series_date, yhat_series_date

    #todo: conferir normalização e ver se dá pra padronizar mais coisas entre os difentes algoritmos
    def sarimax_train_predict(self, train, test,  identifier, cols, in_size, out_size, keep_only, architecture, save_path=None):

        scaler = MinMaxScaler(feature_range=(0, 1))
        column_selector = ColumnSelector(cols)
        #reframer = Reframer(n_in=in_size, n_out=out_size)
        #drop_cols = DropColumns(n_in=in_size, n_out=out_size, n_vars=n_vars, keep_only=keep_only)

        #todo: esse preprocessamento remove os nomes de colunas, talvez seja interessante manter
        #   preciso ver uma lógica para pegar as colunas exógenas
        #   a coluna alvo sempre fica na primeira posição depois do preprocessamento? Onde estou definindo isso?
        preprocess_pipeline = Pipeline(
            [
                ('column_selector', column_selector),
                ('scaler', scaler),
                #('reframer', reframer),
               # ('drop_cols', drop_cols)
            ]
        )

        train_pp = pd.DataFrame(preprocess_pipeline.fit_transform(train), index=train.index, columns=cols)
        test_pp = pd.DataFrame(preprocess_pipeline.transform(test), index=test.index, columns=cols)
        exog_train = train_pp.iloc[:, 1:]
        exog_train = exog_train if len(exog_train.columns) else None
        model = SARIMAX(train_pp[cols[0]], exog=exog_train,
                        order=(1, 0, 1), seasonal_order=(0, 0, 0, 26), trend='ct')
        # fit model
        model_fit = model.fit(disp=False)

        exog_test = test_pp.iloc[:, 1:]
        exog_test = exog_test if len(exog_test.columns) else None

        prediction = model_fit.get_prediction(start=len(train), end= len(train)+ len(test) - 1,
                                              exog=exog_test,
                                              dynamic=False)

        #todo: não denormalizei por que peguei direto do input, mas dar uma olhada nisso
        test_Y = test[cols[0]]
        yhat = prediction.predicted_mean
        pred_conf = prediction.conf_int()
        model_path = ""

        denorm_yhat = np.copy([yhat])

        for i, col in enumerate(denorm_yhat.T):
            denorm_yhat[:, i] = denormalize_with(col, len(cols), scaler, 0)

        test_Y_series_date = pd.Series(test_Y, index=test.index)
        yhat_series_date = pd.Series(denorm_yhat[0], index=test.index)

        return model_path, test_Y_series_date, yhat_series_date, pred_conf

    def prophet_train_predict(self, train, test,  identifier, cols, in_size, out_size, keep_only, architecture, save_path=None):

        scaler = MinMaxScaler(feature_range=(0, 1))
        column_selector = ColumnSelector(cols)
        #reframer = Reframer(n_in=in_size, n_out=out_size)
        #drop_cols = DropColumns(n_in=in_size, n_out=out_size, n_vars=n_vars, keep_only=keep_only)

        preprocess_pipeline = Pipeline(
            [
                ('column_selector', column_selector),
                ('scaler', scaler),
                #('reframer', reframer),
               # ('drop_cols', drop_cols)
            ]
        )

        train_pp = pd.DataFrame(preprocess_pipeline.fit_transform(train), index=train.index, columns=cols)
        test_pp = pd.DataFrame(preprocess_pipeline.transform(test), index=test.index,  columns=cols)

        # Prepare data for Prophet
        prophet_data = train_pp.reset_index().rename(columns={'dt': 'ds', cols[0]: 'y'})
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds']).dt.tz_localize(None)
        exog_train = train_pp.iloc[:, 1:]
        exog_test = test_pp.iloc[:, 1:]

        #exog = exog if len(exog.columns) else None
        model = Prophet()

        if len(exog_train.columns) > 0:
            for col in exog_train.columns:
                model.add_regressor(col)

        model.fit(prophet_data)

        # Forecasting
        future = model.make_future_dataframe(periods=len(test), freq='W', include_history=False)
        if len(exog_test.columns) > 0:
            future = pd.concat([future, exog_test.reset_index(drop=True)[cols[1:]]], axis="columns")

        forecast = model.predict(future)

        # Extract forecasted values
        forecast_values = forecast[['ds', 'yhat']].tail(len(test))

        #todo: não denormalizei por que peguei direto do input, mas dar uma olhada nisso
        test_Y = test[cols[0]]
        yhat = forecast_values['yhat'].values
        #pred_conf = prediction.conf_int()
        model_path = ""

        denorm_yhat = np.copy([yhat])
        for i, col in enumerate(denorm_yhat.T):
            denorm_yhat[:, i] = denormalize_with(col, len(cols), scaler, 0)

        test_Y_series_date = pd.Series(test_Y, index=test.index)
        yhat_series_date = pd.Series(denorm_yhat[0], index=test.index)

        return model_path, test_Y_series_date, yhat_series_date

    def run_crossv(self, data, cols, in_size, out_size, keep_only, architecture, save_path=None, model_id=None, start_offset=None, end_offset=None, train_valid_test: tuple = None, results_path = None):

        if len(cols) > 1:
            exog_cols = cols[1:]

            for col in exog_cols:
                data[col] = data[col].shift(out_size)
                data = data[out_size:]

        if results_path is None:
            results_path = f"pred_ref_{architecture}_{model_id}"

        if architecture == 'simple_lstm_v0' or architecture == 'dia_lstm_v0':
            self.run_crossv_lstm(data, cols, in_size, out_size, keep_only, architecture, save_path, model_id, start_offset, end_offset, train_valid_test, results_path)
        elif architecture == 'sarimax':
            self.run_crossv_sarimax(data, cols, in_size, out_size, keep_only, architecture, save_path, model_id, start_offset, end_offset, train_valid_test,results_path)
        elif architecture == 'prophet':
            self.run_crossv_prophet(data, cols, in_size, out_size, keep_only, architecture, save_path, model_id, start_offset, end_offset, train_valid_test, results_path)

    def run_crossv_lstm(self, data, cols, in_size, out_size, keep_only, architecture, save_path=None, model_id=None,
                   start_offset=None, end_offset=None, train_valid_test: tuple = None, results_path=None):

        # separação dos dados
        tscv_split = TimeSeriesSplit(test_size=out_size, n_splits=10)
        pred_list = []
        ref_list = []
        models = []

        for i_split, (train_index, test_index) in enumerate(tscv_split.split(data)):
            cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]
            # cv_test = pd.concat([cv_train.tail(in_size), cv_test], axis="rows")
            model_path, test_Y, yhat = self.lstm_train_predict(cv_train, cv_test, i_split, cols, in_size, out_size,
                                                               keep_only, architecture, save_path)

            pred_list_local = [str(yhat.index[0])]
            pred_list_local.extend(yhat.values)
            pred_list.append(pred_list_local)

            ref_list_local = [str(test_Y.index[0])]
            ref_list_local.extend(test_Y.values)
            ref_list.append(ref_list_local)

            models.append(model_path)

        pred_df = pd.DataFrame(pred_list)
        pred_df.rename(columns={0: 'dt'}, inplace=True)
        ref_df = pd.DataFrame(ref_list)
        ref_df.rename(columns={0: 'dt'}, inplace=True)
        pred_df.set_index('dt', inplace=True)
        ref_df.set_index('dt', inplace=True)

        self.save_pred_ref(f"{results_path}", pred_df, ref_df, model_id, extra=models)

    def run_crossv_sarimax(self, data, cols, in_size, out_size, keep_only, architecture, save_path=None, model_id=None, start_offset=None, end_offset=None, train_valid_test: tuple = None, results_path=None):
        #separação dos dados
        tscv_split = TimeSeriesSplit(test_size=out_size, n_splits=10)
        pred_list = []
        ref_list = []
        models = []

        for i_split, (train_index, test_index) in enumerate(tscv_split.split(data)):
            cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]
            #cv_test = pd.concat([cv_train.tail(in_size), cv_test], axis="rows")

            model_path, test_Y, yhat, pred_conf = self.sarimax_train_predict(cv_train, cv_test, i_split, cols, in_size, out_size, keep_only, architecture, save_path)
            pred_list_local = [str(yhat.index[0])]
            pred_list_local.extend(yhat.values)
            pred_list.append(pred_list_local)

            ref_list_local = [str(test_Y.index[0])]
            ref_list_local.extend(test_Y.values)
            ref_list.append(ref_list_local)

            models.append(model_path)

        pred_df = pd.DataFrame(pred_list)
        pred_df.rename(columns={0: 'dt'}, inplace=True)
        ref_df = pd.DataFrame(ref_list)
        ref_df.rename(columns={0: 'dt'}, inplace=True)
        pred_df.set_index('dt', inplace=True)
        ref_df.set_index('dt', inplace=True)

        self.save_pred_ref(f"{results_path}", pred_df, ref_df, model_id, extra= models)

    def run_crossv_prophet(self, data, cols, in_size, out_size, keep_only, architecture, save_path=None, model_id=None, start_offset=None, end_offset=None, train_valid_test: tuple = None, results_path=None):
        #separação dos dados
        tscv_split = TimeSeriesSplit(test_size=out_size, n_splits=10)
        pred_list = []
        ref_list = []
        models = []


        for i_split, (train_index, test_index) in enumerate(tscv_split.split(data)):
            cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]
            #cv_test = pd.concat([cv_train.tail(in_size), cv_test], axis="rows")

            model_path, test_Y, yhat = self.prophet_train_predict(cv_train, cv_test, i_split, cols, in_size, out_size, keep_only, architecture, save_path)

            pred_list_local = [str(yhat.index[0])]
            pred_list_local.extend(yhat.values)
            pred_list.append(pred_list_local)

            ref_list_local = [str(test_Y.index[0])]
            ref_list_local.extend(test_Y.values)
            ref_list.append(ref_list_local)

            models.append(model_path)

        pred_df = pd.DataFrame(pred_list)
        pred_df.rename(columns={0: 'dt'}, inplace=True)
        ref_df = pd.DataFrame(ref_list)
        ref_df.rename(columns={0: 'dt'}, inplace=True)
        pred_df.set_index('dt', inplace=True)
        ref_df.set_index('dt', inplace=True)

        self.save_pred_ref(f"{results_path}", pred_df, ref_df, model_id, extra= models)
