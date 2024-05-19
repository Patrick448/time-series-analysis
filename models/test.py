import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler
from custom_transforms.transforms import *


def test_should_denormalize_column_zero():
    data = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4],  [0.25, 0.6, 0.6]])
    original_size = 3
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_column = scaler.fit_transform(data)[:,0]

    result = denormalize_with(scaled_data_column, original_size, scaler, 0)

    assert result.shape == (3,)
    assert np.allclose(result, np.array([0.1, 0.2, 0.25]))


def test_should_denormalize_column_two():
    data = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4],  [0.25, 0.6, 0.6]])
    original_size = 3
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_column = scaler.fit_transform(data)[:,2]

    result = denormalize_with(scaled_data_column, original_size, scaler, 2)

    assert result.shape == (3,)
    assert np.allclose(result, np.array([0.3, 0.4, 0.6]))
