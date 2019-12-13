import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import steam.src.shiyu_users_items_code.config as config
import steam.src.shiyu_users_items_code.plotting as plotting
from steam.src.shiyu_users_items_code.users_items_statistics import iterative_items_statistics


sentiment_dict = {
    'Overwhelmingly Positive': 8,
    'Very Positive': 7,
    'Positive': 6,
    'Mostly Positive': 5,
    'Mixed': 4,
    'Mostly Negative': 3,
    'Negative': 2,
    'Very Negative': 1,
    'Overwhelmingly Negative': 0,
}


class GameItemStat(object):
    def __init__(self):
        self.input_file_path = ""
        self.id_item_dict = None
        self.name_id_dict = None
        self.game_id_purchase_number_order = None
        self.final_game_id_set = None
        self.one_hot_feature_threshold = None
        self.one_hot_feature_dict = None
        self.execute_loop = False
        self.feature_counter = collections.Counter()
        self.sentiment_counter = collections.Counter()
        self.price_list = []
        self.discount_price_list = []
        self.max_item_num = -1
        self.game_id_feature_dict = {}
        self.metascore_list = []

    def hook_prepare(self):
        self.one_hot_feature_threshold = [85, 4100]
        self.max_item_num = 8500

        self.execute_loop = True
        id_item_dict = True
        game_id_feature_dict = True
        final_game_id_set = True
        one_hot_feature_dict = True

        self.input_file_path = config.steam_games_file
        self.item_loader(game_id_feature_dict, id_item_dict, final_game_id_set, one_hot_feature_dict)

    def hook_each_line(self, input_dict):
        self.stat_features_and_price(input_dict)
        pass

    def hook_final(self):
        # self.game_feature_distribution_plot()
        # self.result_saver()
        self.final_game_feature_output()

    def item_loader(
            self, game_id_feature_dict=False, id_item_dict=False, final_game_id_set=False, one_hot_feature_set=False):
        if game_id_feature_dict:
            self.game_id_feature_dict = config.gzip_load(config.game_id_feature_dict)
        if id_item_dict:
            self.id_item_dict = config.gzip_load(config.game_id_name_dict_file)
            self.name_id_dict = {game_name: game_id for game_id, game_name in self.id_item_dict.items()}
        if final_game_id_set:
            self.game_id_purchase_number_order = config.gzip_load(config.game_id_purchase_number_order_dict_file)
            self.final_game_id_set = set(
                [game_id for game_id, index in self.game_id_purchase_number_order.items() if index < self.max_item_num])
        if one_hot_feature_set:
            self.one_hot_feature_dict = config.gzip_load(config.one_hot_feature_dict)

    def stat_features_and_price(self, input_dict):
        def price_or_free(raw_price_content):
            if isinstance(raw_price_content, int):
                return float(raw_price_content)
            elif isinstance(raw_price_content, float):
                return raw_price_content
            elif isinstance(raw_price_content, str):
                if raw_price_content in {
                        'Free to Play', 'Free', 'Play for Free!', 'Free To Play', 'Install Now',
                        'Free Mod', 'Free HITMAN™ Holiday Pack', 'Free to Try', 'Free Movie',
                        'Free to Use'}:
                    return 0
                elif raw_price_content in {'Play WARMACHINE: Tactics Demo', 'Play the Demo', 'Third-party'}:
                    return np.nan
            raise ValueError('Unrecognized price: {}'.format(raw_price_content))

        def meta_score(raw_metascore):
            if isinstance(raw_metascore, int):
                return float(raw_metascore)
            elif isinstance(raw_metascore, float):
                return raw_metascore
            elif isinstance(raw_metascore, str):
                if raw_metascore in {'NA'}:
                    return np.nan
            raise ValueError('Unrecognized meta-score: {}'.format(raw_metascore))

        game_id = None
        game_name_id_dict = self.name_id_dict
        if 'id' in input_dict:
            game_id = input_dict['id']
        else:
            game_name = None
            if 'app_name' in input_dict:
                game_name = input_dict['app_name']
            elif 'title' in input_dict:
                game_name = input_dict['title']
            if game_name is not None and game_name in game_name_id_dict:
                game_id = game_name_id_dict[game_name]
        if game_id is not None and game_id in self.final_game_id_set:
            if 'price' in input_dict:
                game_price = price_or_free(input_dict['price'])
            else:
                game_price = np.nan
            if 'discount_price' in input_dict:
                game_discount_price = price_or_free(input_dict['discount_price'])
            else:
                game_discount_price = game_price
            if 'sentiment' in input_dict:
                raw_game_sentiment = input_dict['sentiment']
                if raw_game_sentiment in sentiment_dict:
                    game_sentiment = sentiment_dict[raw_game_sentiment]
                else:
                    game_sentiment = np.nan
            else:
                game_sentiment = np.nan
            if 'metascore' in input_dict:
                game_metascore = meta_score(input_dict['metascore'])
            else:
                game_metascore = np.nan
            game_tag_list = []
            if 'tags' in input_dict:
                game_tag_list = input_dict['tags']
            game_genre_list = []
            if 'genres' in input_dict:
                game_genre_list = input_dict['genres']
            game_spec_list = []
            if 'specs' in input_dict:
                game_spec_list = input_dict['specs']
            game_feature_list = [*game_tag_list, *game_genre_list, *game_spec_list]
            for feature in game_feature_list:
                self.feature_counter[feature] += 1
            self.sentiment_counter[game_sentiment] += 1
            self.price_list.append(game_price)
            self.discount_price_list.append(game_discount_price)
            self.metascore_list.append(game_metascore)
            self.game_id_feature_dict[game_id] = {
                'game_sentiment': game_sentiment,
                'game_price': game_price,
                'game_discount_price': game_discount_price,
                'game_metascore': game_metascore,
                'game_feature_list': game_feature_list
            }

    def result_saver(self):
        config.gzip_save(self.game_id_feature_dict, config.game_id_feature_dict)
        config.gzip_save(self.one_hot_feature_dict, config.one_hot_feature_dict)

    @staticmethod
    def price_transformer_func(raw_price):
        return np.log(1 + raw_price)

    @staticmethod
    def metascore_transformer_func(raw_metascore):
        return raw_metascore / 20

    def game_feature_distribution_plot(self):
        game_feature_count_sequence = [
            feature_count for feature_name, feature_count in self.feature_counter.most_common()]
        plotting.bar_histogram_plot(game_feature_count_sequence, title='Game feature distribution')
        print(game_feature_count_sequence[:10])
        self.one_hot_feature_dict = {}
        for feature_name, feature_count in self.feature_counter.most_common():
            if self.one_hot_feature_threshold[0] < feature_count < self.one_hot_feature_threshold[1]:
                self.one_hot_feature_dict[feature_name] = 0
            elif feature_count > 4100:
                print(feature_count)
        game_sentiment_dict = {sentiment: count for sentiment, count in self.sentiment_counter.most_common()}
        plotting.bar_plot(game_sentiment_dict, title='Game sentiment')
        price_array = np.array(self.price_list)
        price_array = price_array[np.logical_not(np.isnan(price_array))]
        filtered_price_array = price_array[price_array < 200]
        # print(len(price_array) - len(filtered_price_array))
        sorted_price_array = price_array.copy()
        sorted_price_array.sort()
        # print(sorted_price_array[len(filtered_price_array) - 10:])
        plotting.histogram_plot(
            filtered_price_array, title='Price distribution. Total: {}'.format(len(filtered_price_array)),
            bins=70, y_lim=[0, None])
        plotting.histogram_plot(
            np.log(1 + filtered_price_array),
            title='Log(price) distribution. Total: {}'.format(len(filtered_price_array)),
            bins=100, y_lim=[0, None])
        discount_price_array = np.array(self.discount_price_list)
        discount_price_array = discount_price_array[np.logical_not(np.isnan(discount_price_array))]
        filtered_discount_price_array = discount_price_array[discount_price_array < 200]
        # print(len(discount_price_array) - len(filtered_discount_price_array))
        sorted_discount_price_array = discount_price_array.copy()
        sorted_discount_price_array.sort()
        # print(sorted_discount_price_array[len(filtered_discount_price_array) - 10:])
        plotting.histogram_plot(
            filtered_discount_price_array,
            title='Discount price distribution. Total: {}'.format(len(filtered_discount_price_array)),
            bins=70, y_lim=[0, None])
        metascore_list = np.array(self.metascore_list)
        metascore_list = metascore_list[np.logical_not(np.isnan(metascore_list))]
        plotting.histogram_plot(
            metascore_list,
            title='Metascore distribution. Total: {}'.format(len(metascore_list)),
            bins=100)
        plt.show()

    def final_game_feature_output(self):
        game_id_list = []
        data_dict_list = []
        price_feature_name = 'price'
        price_feature_valid = '{}_valid'.format(price_feature_name)
        discount_price_feature_name = 'discount_{}'.format(price_feature_name)
        discount_price_feature_valid = '{}_valid'.format(discount_price_feature_name)
        metascore_feature_name = 'metascore'
        metascore_feature_valid = '{}_valid'.format(metascore_feature_name)
        sentiment_feature_name = 'sentiment'
        sentiment_feature_valid = '{}_valid'.format(sentiment_feature_name)
        for game_id, game_feature_dict in self.game_id_feature_dict.items():
            current_feature_dict = dict(self.one_hot_feature_dict)
            for one_hot_feature in game_feature_dict['game_feature_list']:
                if one_hot_feature in current_feature_dict:
                    current_feature_dict[one_hot_feature] = 1
            raw_price_value = game_feature_dict['game_price']
            if raw_price_value is np.nan:
                current_feature_dict[price_feature_name] = 0
                current_feature_dict[price_feature_valid] = 0
            else:
                current_feature_dict[price_feature_name] = \
                    self.metascore_transformer_func(raw_price_value)
                current_feature_dict[price_feature_valid] = 1
            raw_discount_price_value = game_feature_dict['game_discount_price']
            if raw_discount_price_value is np.nan:
                current_feature_dict[discount_price_feature_name] = 0
                current_feature_dict[discount_price_feature_valid] = 0
            else:
                current_feature_dict[discount_price_feature_name] = \
                    self.metascore_transformer_func(raw_discount_price_value)
                current_feature_dict[discount_price_feature_valid] = 1
            raw_metascore_value = game_feature_dict['game_metascore']
            if raw_metascore_value is np.nan:
                current_feature_dict[metascore_feature_name] = 0
                current_feature_dict[metascore_feature_valid] = 0
            else:
                current_feature_dict[metascore_feature_name] = self.metascore_transformer_func(raw_metascore_value)
                current_feature_dict[metascore_feature_valid] = 1
            raw_sentiment_value = game_feature_dict['game_sentiment']
            if raw_sentiment_value is np.nan:
                current_feature_dict[sentiment_feature_name] = 0
                current_feature_dict[sentiment_feature_valid] = 0
            else:
                current_feature_dict[sentiment_feature_name] = raw_sentiment_value
                current_feature_dict[sentiment_feature_valid] = 1
            game_id_list.append(game_id)
            data_dict_list.append(current_feature_dict)
        final_data_frame = pd.DataFrame(data_dict_list, index=game_id_list)
        final_data_frame.to_excel(config.final_game_feature_input_df)
        config.gzip_save(game_id_list, config.final_training_data_game_id_list)


def main():
    game_item_stat = GameItemStat()
    iterative_items_statistics(game_item_stat)


if __name__ == '__main__':
    main()