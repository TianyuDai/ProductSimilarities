import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import sklearn.model_selection
from scipy.spatial import distance_matrix

import steam.src.shiyu_users_items_code.config as config
import steam.src.shiyu_users_items_code.plotting as plotting


class WrappedDataLoader:
    def __init__(self, data_loader, func):
        self.data_loader = data_loader
        self.func = func

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        batches = iter(self.data_loader)
        for b in batches:
            yield (self.func(*b))


def coplay_parameter_generation():
    train_data_id_list = config.gzip_load(config.final_train_data_game_id_list)
    test_data_id_list = config.gzip_load(config.final_test_data_game_id_list)
    input_data_file = config.final_game_feature_input_df
    output_data_file = config.final_game_le_similarity_output_df

    # input_data_file = config.train_game_feature_input_df
    # output_data_file = config.train_le_similarity_output_df
    # predict_input_data_file = config.test_game_feature_input_df
    # test_size_ratio = 0.1

    # epochs = 60
    epochs = 100
    batch_size = 30
    # batch_size = 20
    input_feature_num = 164
    output_feature_index_list = [0, 1]
    # nn_hidden_state = [50, 10]
    nn_hidden_state = [80, 25, 10]
    # nn_hidden_state = [50, 16, 6]
    relu_negative_slope = 0.3
    # relu_negative_slope = 0.01
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    optimization_method = torch.optim.Adam
    optimizer_parameter_dict = {
        'lr': 1e-4,
        # 'lr': 3e-5,
        'betas': (0.9, 0.999),
        'weight_decay': 0,
        'amsgrad': False
    }
    loss_func = F.mse_loss
    x_lim = [-0.035, 0.035]
    y_lim = [-0.06, 0.105]
    # x_lim = [-0.06, 0.06]
    # y_lim = [-0.1, 0.11]
    return locals()


def copurchase_parameter_generation():
    train_data_id_list = config.gzip_load(config.final_train_data_game_id_list)
    test_data_id_list = config.gzip_load(config.final_test_data_game_id_list)
    input_data_file = config.final_game_feature_input_df
    output_data_file = config.final_copurchase_similarity_output_df

    # input_data_file = config.train_game_feature_input_df
    # output_data_file = config.train_le_similarity_output_df
    # predict_input_data_file = config.test_game_feature_input_df
    # test_size_ratio = 0.1

    # epochs = 60
    epochs = 130
    batch_size = 30
    # batch_size = 20
    input_feature_num = 164
    output_feature_index_list = [1, 2]
    output_feature_num = len(output_feature_index_list)
    # nn_hidden_state = [50, 10]
    nn_hidden_state = [80, 25, 10]
    # nn_hidden_state = [50, 16, 6]
    relu_negative_slope = 0.05
    # relu_negative_slope = 0.01
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    optimization_method = torch.optim.Adam
    optimizer_parameter_dict = {
        'lr': 2e-4,
        # 'lr': 3e-5,
        'betas': (0.9, 0.999),
        'weight_decay': 0,
        'amsgrad': False
    }
    loss_func = F.mse_loss
    x_lim = [-0.001, 0.004]
    y_lim = [-0.004, 0.006]
    return locals()


def random_train_test_split(test_size_ratio, **other_parameters):
    game_id_list = config.gzip_load(config.final_nn_data_game_id_list)
    test_size = round(len(game_id_list) * test_size_ratio)
    test_index_set = set(np.random.choice(range(len(game_id_list)), test_size, replace=True))
    test_game_id_list = []
    train_game_id_list = []
    for index, game_id in enumerate(game_id_list):
        if index in test_index_set:
            test_game_id_list.append(game_id)
        else:
            train_game_id_list.append(game_id)
    config.gzip_save(test_game_id_list, config.final_test_data_game_id_list)
    config.gzip_save(train_game_id_list, config.final_train_data_game_id_list)


def train_test_data_generation(
        input_data_df, output_data_df, train_data_id_list, test_data_id_list):
    input_train_df = input_data_df.loc[train_data_id_list]
    input_test_df = input_data_df.loc[test_data_id_list]
    output_train_df = output_data_df.loc[train_data_id_list]
    output_test_df = output_data_df.loc[test_data_id_list]
    input_train_array = input_train_df.to_numpy()
    input_test_array = input_test_df.to_numpy()
    output_train_array = output_train_df.to_numpy()
    output_test_array = output_test_df.to_numpy()
    predict_input_data_df = input_test_df
    return input_train_array, input_test_array, output_train_array, output_test_array, predict_input_data_df


def data_preparation(
        input_data_file, output_data_file, batch_size, output_feature_index_list, predict_input_data_file=None,
        test_size_ratio=None, train_data_id_list=None, test_data_id_list=None, **other_parameters):
    input_data_df = config.load_pandas(input_data_file)
    output_data_df = config.load_pandas(output_data_file)
    output_data_df = output_data_df.iloc[:, output_feature_index_list]
    input_data_array = input_data_df.to_numpy()
    output_data_array = output_data_df.to_numpy()
    if predict_input_data_file:
        predict_input_data_df = config.load_pandas(predict_input_data_file)
    else:
        predict_input_data_df = None
    if test_size_ratio is not None:
        (
            input_train_array, input_test_array, output_train_array,
            output_test_array) = sklearn.model_selection.train_test_split(
            input_data_array, output_data_array, test_size=test_size_ratio)
    else:
        (
            input_train_array, input_test_array, output_train_array,
            output_test_array, predict_input_data_df) = train_test_data_generation(
            input_data_df, output_data_df, train_data_id_list, test_data_id_list)
    input_train, input_test, output_train, output_test = map(
        torch.Tensor, (input_train_array, input_test_array, output_train_array, output_test_array))
    train_dataset = TensorDataset(input_train, output_train)
    test_dataset = TensorDataset(input_test, output_test)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size * 2)
    return train_data_loader, test_data_loader, input_data_array, output_data_array, \
           output_train_array, output_test_array, predict_input_data_df


def model_generator(
        input_feature_num, nn_hidden_state, output_feature_num, device, relu_negative_slope, **other_parameters):
    nn_state_num_list = [input_feature_num, *nn_hidden_state, output_feature_num]
    network_list = []
    for layer_index in range(1, len(nn_state_num_list)):
        network_list.append(nn.Linear(nn_state_num_list[layer_index - 1], nn_state_num_list[layer_index]))
        if layer_index is not len(nn_state_num_list) - 1:
            network_list.append(nn.LeakyReLU(negative_slope=relu_negative_slope))
    model = nn.Sequential(*network_list)
    model.to(device)
    return model


def optimizer_generator(model, optimization_method, optimizer_parameter_dict, **other_parameters):
    return optimization_method(model.parameters(), **optimizer_parameter_dict)


def training(
        epochs, model, loss_func, optimizer, train_data_loader, test_data_loader, **other_parameters):
    train_loss_score_list = []
    test_loss_score_list = []
    test_loss = sum(loss_func(model(xb), yb) for xb, yb in test_data_loader) / len(test_data_loader)
    test_loss_score_list.append((0, float(test_loss)))
    print('start', test_loss)
    train_count = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_data_loader:
            pred = model(xb)
            loss = loss_func(pred, yb)
            train_loss_score_list.append((train_count, float(loss)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_count += 1
        model.eval()
        with torch.no_grad():
            test_loss = sum(loss_func(model(xb), yb) for xb, yb in test_data_loader) / len(test_data_loader)

        print(epoch, '{:.6E}'.format(test_loss))
        test_loss_score_list.append((train_count, float(test_loss)))
    return train_loss_score_list, test_loss_score_list


def loss_score_plot(train_score_list, test_score_list, **other_parameters):
    train_data_array = np.array(train_score_list).transpose()
    test_data_array = np.array(test_score_list).transpose()
    data_array_dict = {'Training loss': train_data_array, 'Testing loss': test_data_array}
    plotting.line_plot(data_array_dict, title='Loss value during training')


def training_set_prediction_comparison(
        model, input_data_array, output_data_array, x_lim=None, y_lim=None, **other_parameters):
    color_rgb = np.array([31, 119, 180]) / 255
    alpha_value = 0.2
    color_vector = np.concatenate([color_rgb, [alpha_value]])
    model.eval()
    with torch.no_grad():
        prediction_array = np.array(model(torch.Tensor(input_data_array)))
        if prediction_array.shape[1] == 2:
            plotting.scatter2d_plot(
                output_data_array, marker_size=3, color=color_vector, x_lim=x_lim, y_lim=y_lim, title='Real data')
            plotting.scatter2d_plot(
                prediction_array, marker_size=3, color=color_vector, x_lim=x_lim, y_lim=y_lim, title='Prediction')
        else:
            plotting.scatter3d_plot(
                output_data_array, marker_size=3, color=color_vector, x_lim=x_lim, y_lim=y_lim, title='Real data')
            plotting.scatter3d_plot(
                prediction_array, marker_size=3, color=color_vector, x_lim=x_lim, y_lim=y_lim, title='Prediction')
    return prediction_array


def predictions(model, predict_input_data_df, save_path):
    predict_input_data_array = predict_input_data_df.to_numpy()
    model.eval()
    with torch.no_grad():
        prediction_array = np.array(model(torch.Tensor(predict_input_data_array)))
    prediction_df = pd.DataFrame(prediction_array, index=predict_input_data_df.index)
    prediction_df.to_excel(save_path)


def predict_coordinates_to_similarity(
        file_path, output_train_array, output_test_array, output_prediction_file):
    prediction_df = config.load_pandas(file_path)
    prediction_test_output_array = prediction_df.to_numpy()
    prediction_train_test_similarity_matrix = distance_matrix(output_train_array, prediction_test_output_array)
    prediction_test_similarity_matrix = distance_matrix(output_test_array, prediction_test_output_array)
    np.savez_compressed(
        output_prediction_file,
        prediction_train_test_similarity_matrix=prediction_train_test_similarity_matrix,
        prediction_test_similarity_matrix=prediction_test_similarity_matrix)


def main():
    # parameter_dict = coplay_parameter_generation()
    # prediction_save_path = config.predicted_data_frame_file
    # output_prediction_file = config.predicted_train_test_coplay_matrix_file
    parameter_dict = copurchase_parameter_generation()
    prediction_save_path = config.predicted_copurchase_data_frame_file
    output_prediction_file = config.predicted_train_test_copurchase_matrix_file
    # random_train_test_split(**parameter_dict)
    (
        train_data_loader, test_data_loader, input_data_array, output_data_array,
        output_train_array, output_test_array, predict_input_data_df) = data_preparation(
        **parameter_dict)
    model = model_generator(**parameter_dict)
    optimizer = optimizer_generator(model=model, **parameter_dict)
    train_loss_score_list, test_loss_score_list = training(
        model=model, optimizer=optimizer,
        train_data_loader=train_data_loader, test_data_loader=test_data_loader, **parameter_dict)
    loss_score_plot(train_loss_score_list, test_loss_score_list)
    prediction_array = training_set_prediction_comparison(
        model, input_data_array, output_data_array, **parameter_dict)
    predictions(model, predict_input_data_df, prediction_save_path)
    predict_coordinates_to_similarity(
        prediction_save_path, output_train_array, output_test_array, output_prediction_file)
    plt.show()


if __name__ == '__main__':
    main()
