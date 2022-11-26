import matplotlib.pyplot as plt

def qualityLinePlot(labelsize =12, width = 5, x):
    """
    Creates a publication quality line plot
    """
    plt.rc('font', family = 'serif')
    plt.rc('text', usetex = True)
    plt.rc('xtick', labelsize = labelsize)
    plt.rc('ytick', labelsize = labelsize)
    plt.rc('axes', labelsize = labelsize)   
    height = width / 1.618
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.16, bottom=.2, right=.99, top=.90)
    plt.plot(x)
    # plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'], loc='upper left')
    fig.set_size_inches(width, height)
    plt.savefig('./Graphs/Train_Valiation_Loss.png', dpi=300)
    plt.close()