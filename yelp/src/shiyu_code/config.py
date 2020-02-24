from steam.src.shiyu_users_items_code.config import gzip_save, gzip_load, np_load, load_pandas

data_folder = 'yelp/data'
raw_data_folder = '{}/raw_data'.format(data_folder)


test_business_file = '{}/business_example.json'.format(data_folder)
business_file = '{}/business.json'.format(raw_data_folder)
test_review_file = '{}/review_example.json'.format(data_folder)
review_file = '{}/review.json'.format(raw_data_folder)
test_user_file = '{}/user_example.json'.format(data_folder)
user_file = '{}/user.json'.format(raw_data_folder)

user_id_list_from_review = '{}/user_id_list_from_review.gz'.format(data_folder)
business_id_list_from_review = '{}/business_id_list_from_review.gz'.format(data_folder)
business_id_list_restaurant = '{}/business_id_list_restaurant.gz'.format(data_folder)
user_review_dict = '{}/user_review_dict.gz'.format(data_folder)
business_id_order_index_dict_file = '{}/business_id_order_index_dict.gz'.format(data_folder)
coreview_similarity_matrix_file = '{}/coreview_similarity_matrix.gz'.format(data_folder)


def dict_multi_key(target_dict, key_list):
    return [target_dict[key] for key in key_list]


def dict_copy_keys(original_dict, key_list):
    new_dict = {}
    for key in key_list:
        new_dict[key] = original_dict[key]
    return new_dict

