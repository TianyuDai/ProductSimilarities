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

def corrDistribution(arr, name_1, name_2): 
    plt.figure()
    plt.hist(arr, 50, density=True, range=(-1, 1))
    plt.axvline(x=0, color='black')
    plt.xlabel('Corr(q)')
    plt.ylabel('N')
    plt.xlim(-1, 1)
    # plt.title('The correlation of game similarity between\n'+name_1+' and '+name_2)
    plt.savefig('correlation_'+name_1+'_'+name_2+'.pdf')

if __name__ == '__main__': 
    matrix_seller = np.load('seller_game_similarity.npy')
    matrix_buyer = np.load('buyer_game_similarity.npy')
    matrix_standard = np.load('standard_game_similarity.npy')
    matrix_purchase = np.load('seller_purchase_game_similarity.npy')
    matrix_cop_standard = np.load('co-purchase_game_similarity.npy')

    matrix_list = [matrix_seller, matrix_buyer, matrix_standard, matrix_purchase, matrix_cop_standard]
    name_list = ['seller-centric', 'buyer-centric', 'co-play standard', 'seller-centric with co-purchase input', 'co-purchase standard']
    corr_list = []
    for i in range(len(matrix_list)): 
        for j in range(i+1, len(matrix_list)): 
            corr_list.append(matricesCorrelation(matrix_list[i], matrix_list[j]))
            # corrDistribution(corr, name_list[i], name_list[j])
    
    corr_plot = []
    corr_plot.append(corr_list[3])
    corr_plot.append(corr_list[9])
    corr_plot.append(corr_list[8])
    corr_plot.append(corr_list[1])
    corr_plot.append(corr_list[7])
    corr_plot.append(corr_list[4])

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
    plt.savefig('correlation.pdf')
    """
    spec = np.loadtxt('similarity_tags.txt')
    spec_cop = np.loadtxt('purchaseSimilarity_tags.txt')
    for i in range(len(spec)): 
        spec[i, i] = 0
    corr = matricesCorrelation(spec, spec_cop)
    corrDistribution(corr, 'seller-centric tag', 'seller-centric with co-purchase input tag')
    """
