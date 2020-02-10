import numpy as np
import ast
import pandas as pd
import pickle
import dataset_divider

def userDataReader(infname, outfname): 
    user_data_handler = open(infname, 'r')
    purchase_game_id = []
    for line in user_data_handler: 
        line = ast.literal_eval(line)
        items = line['items']
        user_game_id = []
        for item in items:
            user_game_id.append(item['item_id'])
        purchase_game_id.append(user_game_id)
    with open(outfname, 'wb') as f: 
        pickle.dump(purchase_game_id, f)

def coPurchaseSimilarity(test_game_id, outfname):
    with open(outfname, 'rb') as f: 
        purchase_game_id = pickle.load(f)
    game_similarity = np.zeros((len(test_game_id), len(test_game_id)))
    idk = 0
    for user in purchase_game_id: 
        print(idk)
        for i in range(len(user)): 
            for j in range(i+1, len(user)): 
                game_i = user[i]
                game_j = user[j]
                # print(game_i, game_j)
                if game_i in test_game_id and game_j in test_game_id: 
                    game_similarity[test_game_id.index(game_i), test_game_id.index(game_j)] += 1
        idk += 1
    np.save('co-purchase_game_similarity', game_similarity)

if __name__ == '__main__': 
    infname = '../../data/australian_users_items.json'
    outfname = 'co-purchase_game_id.pkl'
    userDataReader(infname, outfname)
    data_organizer = dataset_divider.DataOrganizer('tags')
    data_organizer.trainTestDivider()
    coPurchaseSimilarity(data_organizer.test_game_id, outfname)
