import json
import numpy as np
import matplotlib.pyplot as plt

from steam.src.shiyu_users_items_code import plotting
from yelp.src.shiyu_code import config
dict_multi_key = config.dict_multi_key
dict_copy_keys = config.dict_copy_keys


def json_loader_with_hook(file_path, hook_obj):
    with open(file_path, encoding='utf-8') as f_in:
        for line in f_in:
            json_obj = json.loads(line)
            hook_obj.json_loader(json_obj)


class ReviewStat(object):
    def __init__(self):
        self.user_review_dict = {}
        self.user_id_list = []
        self.user_id_set = set()
        self.business_id_list = []
        self.business_id_set = set()

    def json_loader(self, json_dict):
        user_id, business_id = dict_multi_key(
            json_dict, ['user_id', 'business_id'])
        new_json_dict = dict_copy_keys(
            json_dict, ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool'])
        if user_id not in self.user_id_set:
            self.user_id_list.append(user_id)
            self.user_id_set.add(user_id)
        if business_id not in self.business_id_set:
            self.business_id_list.append(business_id)
            self.business_id_set.add(business_id)
        if user_id not in self.user_review_dict:
            self.user_review_dict[user_id] = {}
        current_user_dict = self.user_review_dict[user_id]
        if business_id not in current_user_dict:
            current_user_dict[business_id] = []
        current_user_dict[business_id].append(new_json_dict)

    def save_files(self, user_id_collection_file, business_id_collection_file, user_review_dict_file):
        config.gzip_save(self.user_id_list, user_id_collection_file)
        config.gzip_save(self.business_id_list, business_id_collection_file)
        config.gzip_save(self.user_review_dict, user_review_dict_file)


class BusinessStat(object):
    def __init__(self):
        self.restaurant_id_list = []

    def json_loader(self, json_dict):
        business_id, categories_str = dict_multi_key(
            json_dict, ['business_id', 'categories'])
        if categories_str is not None and 'Restaurants' in categories_str:
            self.restaurant_id_list.append(business_id)

    def save_files(self, restaurant_id_list_file):
        config.gzip_save(self.restaurant_id_list, restaurant_id_list_file)


def business_stat_and_restaurant_extract(
        input_file_path, output_restaurant_id_list_file):
    business_stat = BusinessStat()
    json_loader_with_hook(input_file_path, business_stat)
    business_stat.save_files(output_restaurant_id_list_file)


def review_stat_and_classify(
        input_file_path, output_user_id_collection_file, output_business_id_collection_file,
        output_user_review_dict_file):
    review_stat = ReviewStat()
    json_loader_with_hook(input_file_path, review_stat)
    review_stat.save_files(
        output_user_id_collection_file, output_business_id_collection_file, output_user_review_dict_file)


def restaurant_statistics(user_review_dict):
    business_total_review_dict = {}
    business_total_reviewer_dict = {}
    business_total_score_list_dict = {}
    for user_id, user_review_dict in user_review_dict.items():
        for business_id, business_review_list in user_review_dict.items():
            if business_id not in business_total_review_dict:
                business_total_review_dict[business_id] = 0
            business_total_review_dict[business_id] += len(business_review_list)
            if business_id not in business_total_reviewer_dict:
                business_total_reviewer_dict[business_id] = 0
            business_total_reviewer_dict[business_id] += 1
            if business_id not in business_total_score_list_dict:
                business_total_score_list_dict[business_id] = []
            business_total_score_list_dict[business_id].extend(
                [review_dict['stars'] for review_dict in business_review_list])
    sorted_business_review_pair_list = sorted(
        business_total_review_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_business_review_value_list = [value for _, value in sorted_business_review_pair_list]
    sorted_business_index_dict_by_review = {
        business_id: order_index for order_index, (business_id, _) in enumerate(sorted_business_review_pair_list)}
    sorted_business_reviewer_pair_list = sorted(
        business_total_reviewer_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_business_reviewer_value_list = [value for _, value in sorted_business_reviewer_pair_list]
    business_mean_score_dict = {
        business_id: np.mean(score_list) for business_id, score_list in business_total_score_list_dict.items()}
    sorted_business_mean_score_pair_list = sorted(
        business_mean_score_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_business_mean_score_value_list = [value for _, value in sorted_business_mean_score_pair_list]

    config.gzip_save(sorted_business_index_dict_by_review, config.business_id_order_index_dict_file)
    plotting.scatter_bar_plot(sorted_business_review_value_list, title='Review sort')
    plotting.scatter_bar_plot(sorted_business_reviewer_value_list, title='Reviewer sort')
    plotting.scatter_bar_plot(sorted_business_mean_score_value_list, title='Mean score sort')
    plt.show()


def restaurant_similarity_by_review_record(
        user_review_dict, business_id_order_index_dict):
    game_num = len(business_id_order_index_dict)
    raw_coreview_num_matrix = np.zeros([game_num, game_num])
    total_review_num_array = np.ones([game_num])
    total_user_num = len(user_review_dict)
    count = 0
    for user_id, user_review_dict in user_review_dict.items():
        current_review_num = len(user_review_dict)
        business_id_list = list(user_review_dict.keys())
        business_index_list = [business_id_order_index_dict[business_id] for business_id in business_id_list]
        review_num_list = [len(review_list) for review_list in user_review_dict.values()]
        for index_i in range(current_review_num):
            business_index_i = business_index_list[index_i]
            total_review_num_array[business_index_i] += review_num_list[index_i]
            for index_j in range(index_i + 1, current_review_num):
                business_index_j = business_index_list[index_j]
                total_review_num = (review_num_list[index_i] + review_num_list[index_j]) / 2
                raw_coreview_num_matrix[business_index_i, business_index_j] += total_review_num
                raw_coreview_num_matrix[business_index_j, business_index_i] += total_review_num
        count += 1
        if count % 1000 == 0:
            print("{:.3f} finished...".format(count / total_user_num))
    review_sum_matrix = total_review_num_array.reshape([-1, 1]) @ np.ones([1, game_num])
    coreview_similarity_matrix = raw_coreview_num_matrix / (
            review_sum_matrix + review_sum_matrix.T)
    np.savez_compressed(
        config.coreview_similarity_matrix_file, coreview_similarity_matrix=coreview_similarity_matrix)
    plotting.heatmap_plot(copurchase_similarity_matrix)
    plt.show()


def main():
    test = False
    if test:
        review_file = config.test_review_file
        business_file = config.test_business_file
        user_file = config.test_user_file
    else:
        review_file = config.review_file
        business_file = config.business_file
        user_file = config.user_file
    user_review_file = config.user_review_dict
    user_id_list_file = config.user_id_list_from_review
    business_id_list_file = config.business_id_list_from_review
    business_id_list_restaurant_file = config.business_id_list_restaurant

    # review_stat_and_classify(
    #     review_file, user_id_list_file, business_id_list_file, user_review_file)
    business_stat_and_restaurant_extract(business_file, business_id_list_restaurant_file)

    # user_review_dict = config.gzip_load(user_review_file)
    # restaurant_statistics(user_review_dict)
    # restaurant_similarity_by_review_record(user_review_dict)
    print()


if __name__ == '__main__':
    main()
