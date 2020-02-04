import numpy as np
import gzip
import pickle
import ast

class DataOrganizer: 
    def __init__(self, spec): 
        self.train_game_id = []
        self.test_game_id = []
        self.train_game_spec = []
        self.test_game_spec = []
        self.raw_game_id = []
        self.raw_game_spec = []
        self.train_game_id_discard = []
        self.test_game_id_discard = []
        self.n_train = 0
        self.n_test = 0
        self.spec = spec

    def readRawData(self): 
        raw_data_handler = open('../../data/steam_games.json', 'r')
        for line in raw_data_handler: 
            line = ast.literal_eval(line)
            if 'id' in line and self.spec in line: 
                game_id = line['id']
                game_spec = line[self.spec]
                self.raw_game_id.append(game_id)
                self.raw_game_spec.append(game_spec)
        raw_data_handler.close()
            
    def trainTestDivider(self): 
        with gzip.open("../../data/train_game_id_list.gz", 'rb') as train_id_handler: 
            self.train_game_id = pickle.load(train_id_handler)
        
        with gzip.open('../../data/test_game_id_list.gz', 'rb') as test_id_handler: 
            self.test_game_id = pickle.load(test_id_handler)

    def specDivider(self): 
        for game_id in self.train_game_id: 
            # assert game_id in self.raw_game_id, "Train id "+game_id+" is not in the raw data! "
            if game_id in self.raw_game_id: 
                idk = self.raw_game_id.index(game_id)
                game_spec = self.raw_game_spec[idk]
                # assert len(game_spec) != 0, "The spec of id "+game_id+" is empty! "
                self.train_game_spec.append(game_spec)
            else: 
                self.train_game_id_discard.append(game_id)
                # print(len(self.train_game_id)+1)
        for game_id in self.train_game_id_discard: 
            self.train_game_id.remove(game_id)
        self.n_train = len(self.train_game_id)

        for game_id in self.test_game_id: 
            # assert game_id in self.raw_game_id, "Train id "+game_id+" is not in the raw data! "
            if game_id in self.raw_game_id: 
                idk = self.raw_game_id.index(game_id)
                game_spec = self.raw_game_spec[idk]
                # assert len(game_spec) != 0, "The spec of id "+game_id+" is empty! "
                self.test_game_spec.append(self.raw_game_spec[idk])
            else: 
                self.test_game_id_discard.append(game_id)
        for game_id in self.test_game_id_discard: 
            self.test_game_id.remove(game_id)
        self.n_test = len(self.test_game_id)

        # np.save('train_game_id', self.train_game_id)
        # np.save('test_game_id', self.test_game_id)
        # np.save('train_game_spec', self.train_game_spec)
        # np.save('test_game_spec', self.test_game_spec)

if __name__ == '__main__': 
    data_organizer = DataOrganizer('tags') 
    print('read raw data...')
    data_organizer.readRawData()
    print('divide data into training and test set...')
    data_organizer.trainTestDivider()
    data_organizer.specDivider()
    print(data_organizer.n_train)
    print(data_organizer.n_test)
