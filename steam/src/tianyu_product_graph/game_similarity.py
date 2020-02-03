import numpy as np
import dataset_divider
import pandas as pd

class GameSimilarity: 
    def __init__(self, spec, data_organizer): 
        self.spec = spec
        self.data_organizer = data_organizer

    def load_pandas(self, fname): 
        target_file = pd.read_excel(fname, dtype={'id': str})
        target_file.set_index('id', inplace=True)
        return target_file

    def seller_similarity(self, spec_similarity_fname, seller_similarity_fname): 
        spec_stat_file = 'stat_'+self.spec+'.txt'
        spec_idk = {}
        idk = 0
        with open(spec_stat_file, 'r') as fin: 
            for line in fin: 
                spec_name, spec_num = line.rstrip().split('\t')
                spec_idk[spec_name] = idk
                idk += 1

        spec_similarity = np.loadtxt(spec_similarity_fname)
        for i in range(len(spec_similarity)): 
            spec_similarity[i, i] = 1

        game_similarity = np.zeros((self.data_organizer.n_test, self.data_organizer.n_test))
        for i in range(self.data_organizer.n_test): 
            for j in range(self.data_organizer.n_test): 
                game_ij_similarity = 0
                game_i_spec = self.data_organizer.test_game_spec[i]
                game_j_spec = self.data_organizer.test_game_spec[j]
                n_spec_terms = 0
                for spec_i in game_i_spec: 
                    if spec_i in game_j_spec: 
                        game_ij_similarity += 1
                        n_spec_terms += 1
                    else: 
                        for spec_j in game_j_spec: 
                            game_ij_similarity += spec_similarity[spec_idk[spec_i]][spec_idk[spec_j]]
                            n_spec_terms += 1
                game_ij_similarity /= n_spec_terms
                game_similarity[i, j] = game_ij_similarity
        np.save(seller_similarity_fname, game_similarity)

    def buyer_similarity(self, purchase_euclidean_fname, buyer_similarity_fname): 
        game_euclidean = self.load_pandas(purchase_euclidean_fname)
        game_similarity = np.zeros((self.data_organizer.n_test, self.data_organizer.n_test))

        for i, game_i in zip(range(self.data_organizer.n_test), self.data_organizer.test_game_id): 
            for j, game_j in zip(range(self.data_organizer.n_test), self.data_organizer.test_game_id): 
                x_i = game_euclidean.loc[game_i, 0]
                y_i = game_euclidean.loc[game_i, 1]
                x_j = game_euclidean.loc[game_j, 0]
                y_j = game_euclidean.loc[game_j, 1]
                game_similarity[i, j] = np.sqrt((x_i-x_j)**2+(y_i-y_j)**2)

        np.save(buyer_similarity_fname, game_similarity)
        
if __name__ == '__main__': 
    data_organizer = dataset_divider.DataOrganizer('tags')
    data_organizer.readRawData()
    data_organizer.trainTestDivider()
    data_organizer.specDivider()
    
    game_similarity = GameSimilarity('tags', data_organizer)
    game_similarity.seller_similarity('similarity_tags.txt', 'seller_game_similarity')
    game_similarity.seller_similarity('purchaseSimilarity_tags.txt', 'seller_purchase_game_similarity')
    game_similarity.buyer_similarity('../../data/predicted_game_similarity_data_frame.xlsx', 'buyer_game_similarity')
    game_similarity.buyer_similarity('../../data/output_game_similarity_data_frame_for_training.xlsx', 'standard_game_similarity')
