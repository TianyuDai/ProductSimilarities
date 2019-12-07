import collections

import steam.src.shiyu_users_items_code.config as config


class UsersItemsStat(object):
    def __init__(self):
        # self.input_file_path = config.test_users_items_file
        self.input_file_path = config.users_items_file

        self.items_counter = collections.Counter()
        self.items_playtime_2weeks_counter = collections.Counter()
        self.items_playtime_forever_counter = collections.Counter()
        self.game_name_dict = {}
        self.total_game_num = 200

    def hook_each_line(self, input_dict):
        for item in input_dict['items']:
            item_name = item['item_name']
            self.items_counter[item_name] += 1
            self.items_playtime_2weeks_counter[item_name] += item['playtime_2weeks']
            self.items_playtime_forever_counter[item_name] += item['playtime_forever']

    def hook_final(self):
        total_game_num = self.total_game_num
        with open(config.game_stat_file, 'w', encoding='utf-8') as f_out:
            f_out.write("Name, Num of buyers, , Name, 2weeks play time, , Name, Total play time\n")
            for ((item_name, counter_num),
                 (item_name_playtime_2weeks, playtime_2weeks),
                 (item_name_playtime_forever, playtime_forever)) in zip(
                    self.items_counter.most_common(total_game_num),
                    self.items_playtime_2weeks_counter.most_common(total_game_num),
                    self.items_playtime_forever_counter.most_common(total_game_num)):
                f_out.write("{}, {}, , {}, {}, , {}, {}\n".format(
                    item_name, counter_num, item_name_playtime_2weeks, playtime_2weeks,
                    item_name_playtime_forever, playtime_forever))


def example_data_generator():
    def generator_for_one_file(raw_file_name, example_file_name, count):
        with open(raw_file_name, encoding='utf-8') as f_in, open(example_file_name, 'w', encoding='utf-8') as f_out:
            for i in range(count):
                f_out.write(f_in.readline())

    line_num = 100
    generator_for_one_file(config.users_items_file, config.test_users_items_file, line_num)
    generator_for_one_file(config.steam_games_file, config.test_steam_games_file, line_num)


def json_file_loader(fp):
    for line in fp:
        yield eval(line)


def users_items_statistics(operation_obj):
    with open(operation_obj.input_file_path, encoding='utf-8') as f_in:
        for input_dict in json_file_loader(f_in):
            operation_obj.hook_each_line(input_dict)
    operation_obj.hook_final()


def main():
    # example_data_generator()
    users_items_processor = UsersItemsStat()
    users_items_statistics(users_items_processor)


if __name__ == '__main__':
    main()
