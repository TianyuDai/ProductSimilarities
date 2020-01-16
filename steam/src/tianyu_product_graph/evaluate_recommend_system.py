import numpy as np
import ast
import sys
import random
import time
import gzip
import matplotlib.pyplot as plt
sys.path.append('../shiyu_users_items_code')
import config as config

class EvaluateRecommendation: 
    def __init__(self): 
        self.product_idk = {}
        self.product_list = []
        self.n_games = 0
        self.product_graph = []
        self.product_spec = []
        self.game_id_order = {}
        self.product_graph_NN = []

    def productDataReader(self): 
        idk = 0
        # with open('../../data/steam_games_lite.example.json', 'r') as steam_data_handler: 
        steam_data_handler = open('../../data/steam_games_large.json', 'r')
        for line in steam_data_handler: 
            # line = steam_data_handler.readline()
            # while line: 
            line = ast.literal_eval(line)
            # print(line)
            product_name = line['app_name']
            self.product_list.append(product_name)
            self.product_idk[product_name] = idk
            self.product_spec.append(line['tags'])
            idk += 1
        self.n_games = idk
        steam_data_handler.close()

    def productSimilarityDataReader(self): 
        # with gzip.open('../../data/game_id_purchase_number_order.gz') as fin: 
        self.game_id_order = config.gzip_load(config.game_id_purchase_number_order_dict_file)
        self.product_graph_NN = np.load('../../data/game_weighted_similarity_matrix.npz')
        print(self.product_graph_NN[0])

    def readProductGraph(self): 
        self.product_graph = np.loadtxt('product_graph')

    def recommendSystem(self, query_idk, k): 
        sim_product_idk = np.argpartition(self.product_graph[query_idk], -k)[-k:]
        sim_product = [self.product_list[i] for i in sim_product_idk]
        return sim_product

    def positiveBiasEval(self, n_tests, k): 
        same_spec_dist = []
        for i in range(n_tests): 
            random_product = random.randint(0, self.n_games-1)
            random_product_spec = self.product_spec[random_product]
            # print("query product: ")
            # print(self.product_list[random_product], random_product_spec)
            # start_time = time.time()
            sim_product = self.recommendSystem(random_product, k)
            # final_time = time.time()
            # print("recommend time is %s seconds. " %(time.time()-final_time))
            total_sim_spec = []
            n_same_spec = 0
            for sim_item in sim_product: 
                if self.product_idk[sim_item] != random_product: 
                    sim_spec = self.product_spec[self.product_idk[sim_item]]
                    # print(sim_item, sim_spec)
                    for spec_item in sim_spec: 
                       if spec_item in random_product_spec:
                           n_same_spec += 1
            same_spec_dist.append(n_same_spec/len(random_product_spec)/k)
        # print("evaluate time is %s seconds. " %(time.time()-final_time))
        return same_spec_dist

    def posBiasEvalPlot(self, n_tests): 
        k_list = [3, 5, 7, 9]
        n_bins = 10
        plt.figure()
        for k in k_list: 
            same_spec_dist = self.positiveBiasEval(n_tests, k)
            plt.hist(same_spec_dist, n_bins, density=True, histtype='step', stacked='True', fill=False, linewidth=2, label='k = %d' %k)
        plt.legend()
        plt.xlabel("average ratio of tags in a recommended game")
        plt.ylabel("distribution of queries")
        plt.savefig('posBiasEval_Aba.png')

    def proportionalityEval(self, n_tests, k, spec): 
        query_spec = []
        query_sim_spec = []
        for i in range(n_tests): 
            random_product = random.randint(0, self.n_games-1)
            random_product_spec = self.product_spec[random_product]
            for spec_item in random_product_spec: 
                query_spec.append(spec_item)
            sim_product = self.recommendSystem(random_product, k)
            for sim_item in sim_product: 
                sim_spec = self.product_spec[self.product_idk[sim_item]]
                for spec_item in sim_spec: 
                    query_sim_spec.append(spec_item)
        with open("propEval_que", 'w') as fout: 
            x = 0
            for i in spec: 
                x += query_spec.count(i)/len(query_spec)
                fout.write(i+'\t%s\n' %(query_spec.count(i)/len(query_spec)))
            fout.write('others\t%s\n' %(1-x))
        with open("propEval_sim", 'w') as fout: 
            x = 0
            for i in spec: 
                x += query_sim_spec.count(i)/len(query_sim_spec)
                fout.write(i+"\t%s\n" %(query_sim_spec.count(i)/len(query_sim_spec)))
            fout.write('others\t%s\n' %(1-x))
        original_spec_list = []
        original_spec_count = []
        with open('stat_tags.txt', 'r') as fin: 
            for line in fin: 
                original_spec, original_count = line.rstrip().split('\t')
                original_spec_list.append(original_spec)
                original_spec_count.append(eval(original_count))
        with open("propEval_ori", "w") as fout: 
            x = 0
            for i in spec: 
                spec_idk = original_spec_list.index(i)
                x += original_spec_count[spec_idk]/sum(original_spec_count)
                fout.write(i+"\t%s\n" %(original_spec_count[spec_idk]/sum(original_spec_count)))
            fout.write('others\t%s\n' %(1-x))

    def propEvalPlot(self, n_tests, k, spec): 
        self.proportionalityEval(n_tests, k, spec)
        que_spec_name = []
        que_spec_ratio = []
        with open('propEval_que', 'r') as fin: 
            for line in fin: 
                spec_name, spec_ratio = line.rstrip().split('\t')
                que_spec_name.append(spec_name)
                que_spec_ratio.append(spec_ratio)
        sim_spec_name = []
        sim_spec_ratio = []
        with open('propEval_sim', 'r') as fin: 
            for line in fin: 
                spec_name, spec_ratio = line.rstrip().split('\t')
                sim_spec_name.append(spec_name)
                sim_spec_ratio.append(spec_ratio)
        ori_spec_name = []
        ori_spec_ratio = []
        with open('propEval_ori', 'r') as fin: 
            for line in fin: 
                spec_name, spec_ratio = line.rstrip().split('\t')
                ori_spec_name.append(spec_name)
                ori_spec_ratio.append(spec_ratio)
        pie_size = 0.3
        plt.figure()
        cmap = plt.get_cmap("Set3")
        pie_colors = cmap(np.arange(len(spec)+1))
        plt.pie(sim_spec_ratio, labels=que_spec_name, colors=pie_colors, radius = 1, wedgeprops=dict(width=pie_size, edgecolor='w'))
        plt.pie(ori_spec_ratio, colors=pie_colors, radius = 1-pie_size, wedgeprops=dict(width=pie_size, edgecolor='w'))
        plt.pie(que_spec_ratio, colors=pie_colors, radius = 1-2*pie_size, wedgeprops=dict(width=pie_size, edgecolor='w'))
        plt.savefig('propEval_Aba.png')

if __name__ == '__main__': 
    rec_eval = EvaluateRecommendation()
    rec_eval.productSimilarityDataReader()
    #rec_eval.productDataReader()
    #rec_eval.readProductGraph()
    #print("product graph loaded")
    #start_time = time.time()
    # same_spec_dist = rec_eval.positiveBiasEval(100, 3)
    #rec_eval.posBiasEvalPlot(30000)
    # print(same_spec_dist)
    #print("Positive bias evaluation time is %s seconds. " %(time.time()-start_time))
    #final_time = time.time()
    # rec_eval.proportionalityEval(100, 3)
    #spec = ['Action', 'Adventure', 'Casual', 'RPG', 'Indie', 'Free to Play', 'Strategy', 'Simulation', 'Great Soundtrack', 'Singleplayer', 'Multiplayer']
    #rec_eval.propEvalPlot(30000, 5, spec)
    #print("Proportionality evaluation time is %s seconds. " %(time.time()-start_time))
