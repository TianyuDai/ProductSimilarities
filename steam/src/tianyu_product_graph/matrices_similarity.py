import numpy as np
import matplotlib.pyplot as plt

def matricesCorrelation(matrix_1, matrix_2): 
    corr_list = []
    for i in range(len(matrix_1)): 
        vector_1 = matrix_1[i]
        vector_2 = matrix_2[i]
        corr = np.corrcoef(vector_1, vector_2)
        corr_list.append(corr[0, 1])
    return corr_list

def matricesDistance(matrix_1, matrix_2): 
    d = np.linalg.norm(matrix_1-matrix_2)
    return d

def corrDistribution(arr, name): 
    plt.figure()
    plt.hist(arr, 50, density=True, range=(-1, 1))
    plt.axvline(x=0, color='black')
    plt.xlabel('correlation')
    plt.ylabel('N')
    plt.xlim(-1, 1)
    plt.title('The correlation of game similarity between\n'+name)
    plt.savefig('correlation_'+name)

if __name__ == '__main__': 
    matrix_seller = np.load('seller_game_similarity.npy')
    matrix_buyer = np.load('buyer_game_similarity.npy')
    matrix_standard = np.load('standard_game_similarity.npy')
    matrix_purchase = np.load('seller_purchase_game_similarity.npy')
    """
    d_sb = matricesDistance(matrix_seller, matrix_buyer)
    d_st = matricesDistance(matrix_seller, matrix_standard)
    d_bt = matricesDistance(matrix_buyer, matrix_standard)
    d_ps = matricesDistance(matrix_purchase, matrix_seller)
    d_pt = matricesDistance(matrix_purchase, matrix_standard)
    d_pb = matricesDistance(matrix_purchase, matrix_buyer)
    """
    c_sb = matricesCorrelation(matrix_seller, matrix_buyer)
    c_st = matricesCorrelation(matrix_seller, matrix_standard)
    c_bt = matricesCorrelation(matrix_buyer, matrix_standard)
    c_ps = matricesCorrelation(matrix_purchase, matrix_seller)
    c_pt = matricesCorrelation(matrix_purchase, matrix_standard)
    c_pb = matricesCorrelation(matrix_purchase, matrix_buyer)

    # print(d_sb, d_st, d_bt, d_ps, d_pt, d_pb)
    # print(c_sb, c_st, c_bt, c_ps, c_pt, c_pb)
    corr_list = [c_sb, c_st, c_bt, c_ps, c_pt, c_pb]
    name_list = ['seller-centric and buyer-centric', 'seller-centric and co-play standard', 'buyer-centric and co-play standard', 'seller-centric and seller-centric with co-purchase input', 'seller-centric with co-purchase input and co-play standard', 'buyer-centric and seller-centric with co-purchase input']
    for corr, name in zip(corr_list, name_list): 
        corrDistribution(corr, name)
    """
    corrDistribution(c_sb, 'sb')
    corrDistribution(c_st, 'st')
    corrDistribution(c_bt, 'bt')
    corrDistribution(c_ps, 'ps')
    corrDistribution(c_pt, 'pt')
    corrDistribution(c_pb, 'pb')
    """
