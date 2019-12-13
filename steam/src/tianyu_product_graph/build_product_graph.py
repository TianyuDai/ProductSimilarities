import numpy as np
import sys
sys.path.append('../AbaSampleCode')
from parse import parse

class SteamRecommendation: 
    def __init__(self, spec): 
        self.product_idk = {}
        self.product_list = []
        self.n_games = 0
        self.spec = spec

    def productDataReader(self): 
        idk = 0
        steam_data_handler = parse('../../data/steam_games_lite.example.json')
        for line in steam_data_handler:
            product_name = line['app_name']
            self.product_list.append(product_name)
            self.product_idk[product_name] = idk
            idk += 1
        self.n_games = idk

    def generateProductGraph(self): 
        spec_stat_file = 'stat_'+self.spec+'.txt'
        spec_idk = {}
        idk = 0
        with open(spec_stat_file, 'r') as fin: 
            for line in fin: 
                spec_name, spec_num = line.rstrip().split('\t')
                spec_idk[spec_name] = idk
                idk += 1
        similarity_graph = np.loadtxt('similarity_'+self.spec+'.txt')
        steam_data_handler = parse('../../data/steam_games_lite.example.json')
        product_graph = np.zeros((self.n_games, self.n_games))
        for line_1 in steam_data_handler: 
            product_1 = self.product_idk[line_1['app_name']]
            spec_values_1 = line_1[self.spec]
            for line_2 in steam_data_handler: 
                product_2 = self.product_idk[line_2['app_name']]
                spec_values_2 = line_2[self.spec]
                product_1v2_similarity = 0
                for i in spec_values_1: 
                    for j in spec_values_2: 
                        product_1v2_similarity += similarity_graph[spec_idk[i]][spec_idk[j]]
                product_1v2_similarity /= len(spec_values_1)*len(spec_values_2)
                product_graph[product_1][product_2] = product_graph[product_2][product_1] = product_1v2_similarity

        np.savetxt('product_graph', product_graph)

    def findSimilarProduct(self, query, k): 
        product_graph = np.loadtxt('product_graph')
        query_idk = self.product_idk[query]
        query_similarity = product_graph[query_idk]
        sim_product_idk = np.argpartition(query_similarity, -k)[-k:]
        sim_product = [self.product_list[i] for i in sim_product_idk]
        print("The recommended %d products: " %k)
        for i in sim_product: 
            print(i)

if __name__ == '__main__': 
    rec_tags = SteamRecommendation('tags')
    rec_tags.productDataReader()
    rec_tags.generateProductGraph()
    query = ''
    while(query not in rec_tags.product_list): 
        query = input("Please input the name of the game: ")
        if query not in rec_tags.product_list: 
            print(query+" is not in the dataset! Please choose another game. ")
    rec_tags.findSimilarProduct(query, 5)
