import numpy as np
import sys
import ast
import time

class SteamRecommendation: 
    def __init__(self, spec, data_path): 
        self.product_idk = {}
        self.product_list = []
        self.n_games = 0
        self.spec = spec
        self.data_path = data_path

    def productDataReader(self): 
        idk = 0
        steam_data_handler = open(self.data_path, 'r')
        for line in steam_data_handler: 
            line = ast.literal_eval(line)
            product_name = line['app_name']
            self.product_list.append(product_name)
            self.product_idk[product_name] = idk
            idk += 1
        self.n_games = idk
        steam_data_handler.close()

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
        #steam_data_handler = parse('../../data/steam_games_large.json')
        product_graph = np.zeros((self.n_games, self.n_games))
        pointer_ini = 0
        with open(self.data_path, 'r') as steam_data_handler:
            line_1 = steam_data_handler.readline()
            while line_1:
                line_1 = ast.literal_eval(line_1)
                pointer_loc = steam_data_handler.tell()
                product_1 = self.product_idk[line_1['app_name']]
                print(product_1)
                spec_values_1 = line_1[self.spec]
                steam_data_handler.seek(pointer_ini, 0)
                pointer_ini = pointer_loc
                line_2 = steam_data_handler.readline()
                while line_2: 
                    line_2 = ast.literal_eval(line_2)
                    product_2 = self.product_idk[line_2['app_name']]
                    spec_values_2 = line_2[self.spec]
                    product_1v2_similarity = 0
                    n_spec_terms = 0
                    for i in spec_values_1: 
                        if i in spec_values_2: 
                            product_1v2_similarity += 1
                            n_spec_terms += 1
                        else: 
                            for j in spec_values_2: 
                                product_1v2_similarity += similarity_graph[spec_idk[i]][spec_idk[j]]
                                n_spec_terms += 1
                    product_1v2_similarity /= n_spec_terms
                    product_graph[product_1][product_2] = product_graph[product_2][product_1] = product_1v2_similarity
                    line_2 = steam_data_handler.readline()
                steam_data_handler.seek(pointer_loc, 0)
                line_1 = steam_data_handler.readline()
        np.savetxt('product_graph_small', product_graph)

    def findSimilarProduct(self, query, k): 
        start_time = time.time()
        product_graph = np.loadtxt('product_graph_small')
        print("product graph loaded, time = %s seconds" %(time.time()-start_time))
        final_time = time.time()
        query_idk = self.product_idk[query]
        query_similarity = product_graph[query_idk]
        print("find the recommended games...")
        sim_product_idk = np.argpartition(query_similarity, -k)[-k:]
        sim_product = [self.product_list[i] for i in sim_product_idk]
        print("The recommended %d products: " %k)
        for i in sim_product: 
            print(i)
        print("time of finding the product is %s seconds" %(time.time()-final_time))

if __name__ == '__main__': 
    rec_tags = SteamRecommendation('tags', '../../data/steam_games_lite.example.json')
    print("read data...")
    rec_tags.productDataReader()
    # print("generate product graph")
    rec_tags.generateProductGraph()
    query = ''
    while(query not in rec_tags.product_list): 
        query = input("Please input the name of the game: ")
        if query not in rec_tags.product_list: 
            print(query+" is not in the dataset! Please choose another game. ")
    rec_tags.findSimilarProduct(query, 5)
