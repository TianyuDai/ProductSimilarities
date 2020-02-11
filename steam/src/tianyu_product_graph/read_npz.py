import numpy as np
import gzip
import pickle
import dataset_divider

def readSimilarityFiles(fname): 
    game_similarity_list = np.load(fname, 'r')
    for k in game_similarity_list: 
        print(k)
    game_similarity = game_similarity_list['test_similarity_matrix']
    for i in range(len(game_similarity)): 
    #     print(np.max(game_similarity[i]))
        print(game_similarity[i, i])
    np.save('shiyu_coplay_similarity', game_similarity)

def testValideGames(data_organizer): 
    test_game_id = data_organizer.test_game_id
    with gzip.open("../../data/test_game_id_list.gz", 'rb') as train_id_handler:
        shiyu_test_game_id = pickle.load(train_id_handler)
    print(len(test_game_id), len(shiyu_test_game_id))
    for i in range(len(shiyu_test_game_id)): 
        print(shiyu_test_game_id[i], test_game_id[i])

if __name__ == '__main__': 
    data_organizer = dataset_divider.DataOrganizer('tags') 
    data_organizer.readRawData() 
    data_organizer.trainTestDivider()
    data_organizer.specDivider()

    # testValideGames(data_organizer)
    readSimilarityFiles('train_test_similarity_matrix.npz')
