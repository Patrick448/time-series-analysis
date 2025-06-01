import math

def train_test_validation_split(data, train_percentage=None, test_percentage=None, train_valid_test: tuple = None):
    dataset_size = len(data)
    if train_valid_test is None:
        train_size = math.ceil(dataset_size * train_percentage)
        test_size = math.ceil(dataset_size * test_percentage)
        validation_size = dataset_size - train_size - test_size
    else:
        train_size =train_valid_test[0]
        validation_size = train_valid_test[1]
        test_size = train_valid_test[2]

    print(f"train_size: {train_size}\nvalidation_size: {validation_size}\ntest_size: {test_size}")

    train = data[0:train_size]
    validation = data[train_size:train_size + validation_size]
    test = data[train_size + validation_size:dataset_size]
    print(
        f"values_train: {train.shape}\nvalues_validation: {validation.shape}\nvalues_test: {test.shape}")
    # print final percentages
    train_percentage = len(train) / dataset_size
    validation_percentage = len(validation) / dataset_size
    test_percentage = len(test) / dataset_size
    print(
        f"train_percentage: {train_percentage}\nvalidation_percentage: {validation_percentage}\ntest_percentage: {test_percentage}")
    return train, validation, test


def input_output_split(data, in_size, out_size):
    #split into input and output
    train_X, train_Y = data.values[:, :-out_size], data.values[:, -out_size:]
    return train_X, train_Y