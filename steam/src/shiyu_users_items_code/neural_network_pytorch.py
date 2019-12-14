import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import sklearn.model_selection

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


def parameter_generation():
    input_data_file = config.final_game_feature_input_df
    output_data_file = config.final_game_le_similarity_output_df
    test_size_ratio = 0.2

    epochs = 20
    batch_size = 30
    input_feature_num = 164
    output_feature_num = 3
    nn_hidden_state = [50, 10]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    optimization_method = torch.optim.Adam
    optimizer_parameter_dict = {
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'weight_decay': 0,
        'amsgrad': False
    }
    loss_func = F.mse_loss
    return locals()


def data_preparation(
        input_data_file, output_data_file, batch_size, test_size_ratio, output_feature_num, **other_parameters):
    input_data_df = pd.read_excel(input_data_file)
    output_data_df = pd.read_excel(output_data_file)
    if output_feature_num < 3:
        output_data_df = output_data_df.iloc[:, :output_feature_num]
    input_data_array = input_data_df.to_numpy()[:, 1:]
    output_data_array = output_data_df.to_numpy()[:, 1:]
    input_train, input_test, output_train, output_test = sklearn.model_selection.train_test_split(
        input_data_array, output_data_array, test_size=test_size_ratio)
    input_train, input_test, output_train, output_test = map(
        torch.Tensor, (input_train, input_test, output_train, output_test))
    train_dataset = TensorDataset(input_train, output_train)
    test_dataset = TensorDataset(input_test, output_test)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size * 2)
    return train_data_loader, test_data_loader


def model_generator(input_feature_num, nn_hidden_state, output_feature_num, device, **other_parameters):
    nn_state_num_list = [input_feature_num, *nn_hidden_state, output_feature_num]
    network_list = []
    for layer_index in range(1, len(nn_state_num_list)):
        network_list.append(nn.Linear(nn_state_num_list[layer_index - 1], nn_state_num_list[layer_index]))
        network_list.append(nn.LeakyReLU())
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
    plt.show()


def main():
    parameter_dict = parameter_generation()
    train_data_loader, test_data_loader = data_preparation(**parameter_dict)
    model = model_generator(**parameter_dict)
    optimizer = optimizer_generator(model=model, **parameter_dict)
    train_loss_score_list, test_loss_score_list = training(
        model=model, optimizer=optimizer,
        train_data_loader=train_data_loader, test_data_loader=test_data_loader, **parameter_dict)
    loss_score_plot(train_loss_score_list, test_loss_score_list)


if __name__ == '__main__':
    main()
