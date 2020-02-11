import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, pearsonr

def matricesCorrelation(matrix_1, matrix_2): 
    corr_list = []
    # print(matrix_1.shape)
    for i in range(len(matrix_1)): 
        v_1 = matrix_1[i]
        v_2 = matrix_2[i]
        # print(v_1.shape, matrix_1[i].shape)
        vector_1 = np.argsort(np.argsort(v_1))
        vector_2 = np.argsort(np.argsort(v_2))
        # vector_1 = rankdata(v_1)
        # vector_2 = rankdata(v_2)
        corr = np.corrcoef(vector_1, vector_2)
        # r, _ = pearsonr(vector_1, vector_2)
        corr_list.append(corr[0, 1])
        # corr_list.append(r)
        # print(r)
    return corr_list

def matricesDistance(matrix_1, matrix_2): 
    d = np.linalg.norm(matrix_1-matrix_2)
    return d

def corrDistribution(arr, name_1, name_2): 
    plt.figure()
    plt.hist(arr, 50, density=True, range=(-1, 1))
    plt.axvline(x=0, color='black')
    plt.xlabel('Corr(q)')
    plt.ylabel('N')
    plt.xlim(-1, 1)
    # plt.title('The correlation of game similarity between\n'+name_1+' and '+name_2)
    plt.savefig('correlation_'+name_1+'_'+name_2)

if __name__ == '__main__':
    
    matrix_seller = np.load('seller_game_similarity.npy')
    matrix_seller_copurchase = np.load('seller_purchase_game_similarity.npy')
    matrix_buyer_copurchase = np.load('shiyu_buyer_copurchase_similarity.npy')
    matrix_buyer_coplay = np.load('shiyu_buyer_coplay_similarity.npy')
    matrix_copurchase = np.load('shiyu_copurchase_similarity.npy')
    matrix_coplay = np.load('shiyu_coplay_similarity.npy')
    # matrix_cop_standard = np.load('co-purchase_game_similarity.npy')
    """ 
    for i in range(len(matrix_seller)): 
        # print(matrix_seller[i, i], matrix_seller_copurchase[i, i])
        matrix_seller[i, i] = 0
        matrix_seller_copurchase[i, i] = 0 
        
    matrix_list = [matrix_seller, matrix_seller_copurchase, matrix_buyer_copurchase, matrix_buyer_coplay, matrix_copurchase, matrix_coplay]
    name_list = ['seller-centric', 'buyer-centric', 'co-play standard', 'seller-centric with co-purchase input', 'co-purchase standard']
    corr_list = []
    for i in range(len(matrix_list)): 
        for j in range(i+1, len(matrix_list)): 
            corr_list.append(matricesCorrelation(matrix_list[i], matrix_list[j]))
            # corrDistribution(corr, name_list[i], name_list[j])
    
    corr_plot = []
    corr_plot.append(corr_list[3])
    corr_plot.append(corr_list[7])
    corr_plot.append(corr_list[10])
    corr_plot.append(corr_list[14])
    corr_plot.append(corr_list[4])
    corr_plot.append(corr_list[13])

    num_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(16, 8))
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)
    for corr, ax, num in zip(corr_plot, axs.ravel(), num_list): 
        ax.hist(corr, 50, density=True, range=(-1, 1))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(0, 6.5)
        ax.text(-1.1, 6, num, size=16)
        if (num == num_list[3] or num == num_list[4] or num == num_list[5]): 
            ax.set_xlabel('correlation', fontsize=16)
        if (num == num_list[0] or num == num_list[3]): 
            ax.set_ylabel('N', fontsize=16)
            ax.tick_params(direction="in", which='both', labelsize=16)
        else: 
            ax.tick_params(direction="in", which='both', labelsize=16, labelleft=False)
        ax.axvline(x=0, color='black')
    plt.savefig('correlation')
    """
    
    data = np.load('train_test_copurchase_similarity_matrix.npz', 'r')
    pred = np.load('predicted_train_test_copurchase_similarity_matrix.npz', 'r')
    corr = matricesCorrelation(data['test_similarity_matrix'], pred['prediction_test_similarity_matrix'])
    # corrDistribution(corr, 'coplay', 'embedded')
    
    plt.figure()
    plt.hist(corr, 50, density=True, range=(-1, 1))
    plt.axvline(x=0, color='black')
    plt.xlabel('Corr(q)')
    plt.ylabel('N')
    plt.xlim(-1, 1)
    # plt.title('The correlation of game similarity between\n'+name_1+' and '+name_2)
    plt.savefig('correlation_buyer_copurchase.pdf')
    
