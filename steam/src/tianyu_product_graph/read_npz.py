import numpy as np
import gzip
import pickle
import dataset_divider

def readSimilarityFiles(infname, matname, outfname): 
    game_similarity_list = np.load(infname, 'r')
    # for k in game_similarity_list: 
    #    print(k)
    game_similarity = game_similarity_list[matname]
    # for i in range(len(game_similarity)): 
    #     print(np.max(game_similarity[i]))
    #     print(game_similarity[i, i])
    np.save(outfname, game_similarity)

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
    readSimilarityFiles('train_test_similarity_matrix.npz', 'test_similarity_matrix', 'shiyu_coplay_similarity')
    readSimilarityFiles('train_test_copurchase_similarity_matrix.npz', 'test_similarity_matrix', 'shiyu_copurchase_similarity')
    readSimilarityFiles('predicted_train_test_coplay_similarity_matrix.npz', 'prediction_test_similarity_matrix', 'shiyu_buyer_coplay_similarity')
    readSimilarityFiles('predicted_train_test_copurchase_similarity_matrix.npz', 'prediction_test_similarity_matrix', 'shiyu_buyer_copurchase_similarity')
