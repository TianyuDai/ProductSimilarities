import pickle
import gzip
import numpy as np
import pandas as pd

data_folder = 'steam/data'
raw_data_folder = '{}/raw_data'.format(data_folder)

test_users_items_file = '{}/australian_users_items.example.json'.format(data_folder)
users_items_file = '{}/australian_users_items.json'.format(raw_data_folder)
test_steam_games_file = '{}/steam_games.example.json'.format(data_folder)
steam_games_file = '{}/steam_games.json'.format(raw_data_folder)

game_stat_file = '{}/game_stat_files.xlsx'.format(data_folder)
ignored_games_list_file = '{}/ignored_and_merged_games.xlsx'.format(data_folder)
ignored_class_beta_version = 'beta_version'
ignored_class_low_playtime = 'low_playtime'
complete_game_id_list_file = "{}/complete_game_id_list.gz".format(data_folder)
game_id_name_dict_file = '{}/game_id_name_dict.gz'.format(data_folder)
game_id_purchase_number_order_dict_file = '{}/game_id_purchase_number_order.gz'.format(data_folder)
game_player_playtime_dict_file = '{}/game_player_playtime_dict.gz'.format(data_folder)
player_game_playtime_dict_file = '{}/player_game_playtime_dict.gz'.format(data_folder)
player_weight_dict_file = '{}/player_weight_dict.gz'.format(data_folder)
copurchase_similarity_matrix_file = '{}/copurchase_similarity_matrix.npz'.format(data_folder)
game_weighted_similarity_matrix_file = '{}/game_weighted_similarity_matrix.npz'.format(data_folder)
# train_similarity_matrix_file = '{}/train_similarity_matrix.npz'.format(data_folder)
train_test_similarity_matrix_file = '{}/train_test_similarity_matrix.npz'.format(data_folder)
train_test_copurchase_similarity_matrix_file = '{}/train_test_copurchase_similarity_matrix.npz'.format(data_folder)
predicted_train_test_coplay_matrix_file = '{}/predicted_train_test_coplay_similarity_matrix.npz'.format(data_folder)
predicted_train_test_copurchase_matrix_file = '{}/predicted_train_test_copurchase_similarity_matrix.npz'.format(data_folder)
# test_similarity_matrix_file = '{}/test_similarity_matrix.npz'.format(data_folder)
complete_cluster_labels_array = '{}/complete_cluster_labels.npz'.format(data_folder)

game_id_label = 'ID'
game_name_label = 'Name'
game_buyers_label = 'Num of buyers'
game_two_weeks_playtime_label = '2weeks play time'
game_total_playtime_label = 'Total play time'
game_average_two_weeks_playtime_label = "{} per person".format(game_two_weeks_playtime_label)
game_average_total_playtime_label = "{} per person".format(game_total_playtime_label)

embedded_coordinates_laplacian_eigenmaps = "{}/embedded_coordinates_laplacian".format(data_folder)
embedded_train_coordinates = "{}/embedded_train_coordinates".format(data_folder)
embedded_copurchase_coordinates = "{}/embedded_copurchase_coordinates".format(data_folder)
output_data_matrix_name = 'transformed_coordinates'
parameter_data_file = '{}/parameter_data_dict.gz'.format(data_folder)
final_game_le_similarity_output_df = "{}/output_game_similarity_data_frame_for_training.xlsx".format(data_folder)
final_copurchase_similarity_output_df = "{}/output_copurchase_similarity_data_frame_for_training.xlsx".format(data_folder)
train_le_similarity_output_df = "{}/output_train_game_similarity_data_frame.xlsx".format(data_folder)
train_game_feature_input_df = '{}/input_train_game_feature_data_frame.xlsx'.format(data_folder)
test_game_feature_input_df = '{}/input_test_game_feature_data_frame.xlsx'.format(data_folder)

game_genre_stat = '{}/stat_genres.txt'.format(data_folder)
game_spec_stat = '{}/stat_specs.txt'.format(data_folder)
game_tag_stat = '{}/stat_tags.txt'.format(data_folder)
game_id_feature_dict = '{}/game_id_feature_dict.gz'.format(data_folder)
one_hot_feature_dict = '{}/one_hot_feature_dict.gz'.format(data_folder)
final_game_feature_input_df = '{}/input_game_feature_data_frame_for_training.xlsx'.format(data_folder)
final_nn_data_game_id_list = '{}/game_id_list.gz'.format(data_folder)
final_train_data_game_id_list = '{}/train_game_id_list.gz'.format(data_folder)
final_test_data_game_id_list = '{}/test_game_id_list.gz'.format(data_folder)
predicted_coplay_data_frame_file = '{}/predicted_game_similarity_data_frame.xlsx'.format(data_folder)
predicted_copurchase_data_frame_file = '{}/predicted_copurchase_similarity_data_frame.xlsx'.format(data_folder)


def gzip_save(obj, file_name):
    with gzip.open(file_name, 'w') as f_out:
        pickle.dump(obj, f_out)


def gzip_load(file_name):
    with gzip.open(file_name) as f_in:
        return pickle.load(f_in)


def np_load(file_name, obj_name):
    obj_dict = np.load(file_name)
    if isinstance(obj_name, str):
        return obj_dict[obj_name]
    else:
        return [obj_dict[obj] for obj in obj_name]


def load_pandas(file_name):
    target_df = pd.read_excel(file_name, dtype={'id': str})
    target_df.set_index('id', inplace=True)
    return target_df

