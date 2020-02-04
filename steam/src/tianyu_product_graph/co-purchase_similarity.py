import numpy as np
import ast
import dataset_divider

def userDataReader(fname): 
    user_data_handler = open(fname, 'r')
    purchase_game_id = []
    for line in user_data_handler: 
        line = ast.literal_eval(line)
        items = line['items']
        user_game_id = []
        for item in items:
            user_game_id.append(item['item_id'])
        purchase_game_id.append(user_game_id)
    # print(purchase_game_id)
    np.savetxt('co-purchase_game_id.txt', purchase_game_id, fmt='%s')

def coPurchaseSimilarity(test_game_id):
    purchase_game_id = open('co-purchase_game_id.txt')
    # purchase_game_id = np.genfromtxt('co-purchase_game_id.txt', dtype='str')
    game_similarity = np.zeros((len(test_game_id), len(test_game_id)))
    for user in purchase_game_id: 
        for i in range(len(user)): 
            for j in range(i+1, len(user)): 
                game_i = user[i]
                game_j = user[j]
                if game_i in test_game_id and game_j in test_game_id: 
                    game_similarity[test_game_id.index(game_i), test_game_id.index(game_j)] += 1
    np.save(game_similarity, 'co-purchase_game_similarity')

if __name__ == '__main__': 
    fname = '../../data/australian_users_items.example.json'
    userDataReader(fname)
    data_organizer = dataset_divider.DataOrganizer('tags')
    data_organizer.trainTestDivider()
    coPurchaseSimilarity(data_organizer.test_game_id)
