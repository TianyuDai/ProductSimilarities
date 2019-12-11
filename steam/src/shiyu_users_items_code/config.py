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
game_id_name_dict_file = '{}/game_id_name_dict.gz'.format(data_folder)
game_id_purchase_number_order_dict_file = '{}/game_id_purchase_number_order.gz'.format(data_folder)
game_player_playtime_dict_file = '{}/game_player_playtime_dict.gz'.format(data_folder)
player_game_playtime_dict_file = '{}/player_game_playtime_dict.gz'.format(data_folder)
player_weight_dict_file = '{}/player_weight_dict.gz'.format(data_folder)
game_weighted_similarity_matrix = '{}/game_weighted_similarity_matrix.npz'.format(data_folder)

game_id_label = 'ID'
game_name_label = 'Name'
game_buyers_label = 'Num of buyers'
game_two_weeks_playtime_label = '2weeks play time'
game_total_playtime_label = 'Total play time'
game_average_two_weeks_playtime_label = "{} per person".format(game_two_weeks_playtime_label)
game_average_total_playtime_label = "{} per person".format(game_total_playtime_label)

