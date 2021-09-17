import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

from sklearn.metrics import mean_squared_error;
from constants import *;

multi_data = pd.read_csv('../data/zri_multifamily_v2.csv');

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time Frame")
    plt.ylabel("ZRI")
    plt.grid(True)

def NN_model(dataset):
    tf.keras.backend.clear_session()
    # dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                          input_shape=[None]),
    #   tf.keras.layers.Conv1D(filters=16, kernel_size=3,
    #                       strides=1, padding="causal",
    #                       activation="relu",
    #                       input_shape=[None, 1]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
        #  tf.keras.layers.SimpleRNN(16, return_sequences=True),
        #  tf.keras.layers.SimpleRNN(16, return_sequences=True),
    #   tf.keras.layers.Dense(16, activation="relu"),
    #   tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 2.0)
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=3e-4, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer="adam",
                  metrics=["mae", "mse"])
    history = model.fit(dataset, epochs=500, verbose = 0);
    return model;

def NN_forecast(model, single_city_series):
    forecast = []
    results = []
    for time in range(len(single_city_series) - WINDOW_SIZE):
        forecast.append(model.predict(single_city_series[time:time + WINDOW_SIZE][np.newaxis]))

    #print(forecast)

    # forecast = forecast[SPLIT - WINDOW_SIZE:]
    results = np.array(forecast)[:, 0, 0]
    actual = single_city_series[WINDOW_SIZE:]
    time_actual = range(WINDOW_SIZE, len(single_city_series));

    pure_forecast = list(single_city_series[SPLIT - WINDOW_SIZE: SPLIT]);
    for time in range(SPLIT, len(single_city_series)):
        # print(model.predict(pure_forecast[-WINDOW_SIZE:][np.newaxis]))
        pure_forecast.append(np.array(model.predict(np.array(pure_forecast[-WINDOW_SIZE:])[np.newaxis]))[0][0][0])
    pure_forecast = np.array(pure_forecast[WINDOW_SIZE:]);

    return results, actual, pure_forecast;

@tf.autograph.experimental.do_not_convert
def NN_test(ZONE, plot=False):
    '''
    Input: ZONE
    Output: the RMSE of a NN model on the predicted train, partially predicted test, and complete predicted test.
    '''
    # Collection of data
    single_city_data = multi_data[multi_data["zip"] == ZONE];
    single_city_series = np.array(single_city_data["zri"]);
    
    # Standardization
    single_city_series_mean = single_city_series.mean();
    single_city_series_std = single_city_series.std();

    single_city_series = (single_city_series - single_city_series_mean)/\
    single_city_series_std;
    
    # Train test split
    single_city_train = single_city_series[:SPLIT];
    single_city_test = single_city_series[SPLIT:];
    
    # Window the training set to make input of the NN
    dataset = windowed_dataset(single_city_train, WINDOW_SIZE, BATCH_SIZE, 60);
    model = NN_model(dataset);
    
    time_train = list(range(SPLIT));
    time_test = list(range(SPLIT, len(single_city_series)));
    
    # Forecasting
    results, actual, pure_forecast = NN_forecast(model, single_city_series);
    
    # Compute MSE
    MSE_train = mean_squared_error(actual[:-TEST_LENGTH], results[:-TEST_LENGTH])**0.5 * single_city_series_std;
    MSE_test = mean_squared_error(actual[-TEST_LENGTH:], results[-TEST_LENGTH:])**0.5 * single_city_series_std;
    MSE_pure = mean_squared_error(actual[-TEST_LENGTH:], pure_forecast[-TEST_LENGTH:])**0.5 * single_city_series_std;
    
    if plot: # If the plot option is selected, plot the graph.
        time_actual = range(WINDOW_SIZE, len(single_city_series));
        plt.figure(figsize=(10, 6))
        plot_series(time_actual, actual);
        plot_series(time_actual, results);
        plot_series(time_test, pure_forecast);

    return MSE_train, MSE_test, MSE_pure, np.array(pure_forecast[-TEST_LENGTH:]) * single_city_series_std + single_city_series_mean;