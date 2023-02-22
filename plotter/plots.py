import matplotlib.pyplot as plt


def qualityLinePlot(history, labelsize=12, width=5):
    """
    Creates a publication quality line plot
    """
    plt.rc("font", family="serif")
    plt.rc("xtick", labelsize=labelsize)
    plt.rc("ytick", labelsize=labelsize)
    plt.rc("axes", labelsize=labelsize)

    height = width / 1.618
    fig1, ax1 = plt.subplots()
    fig1.subplots_adjust(left=0.16, bottom=0.2, right=0.99, top=0.90)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"], loc="upper left")
    fig1.set_size_inches(width, height)
    plt.savefig("./Graphs/Train_Valiation_Loss.png", dpi=150)
    plt.close()

    fig2, ax2 = plt.subplots()
    fig2.subplots_adjust(left=0.16, bottom=0.2, right=0.99, top=0.97)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Validation"], loc="upper right")
    fig2.set_size_inches(width, height)
    plt.savefig("./Graphs/Train_Valiation_Accuracy.png", dpi=150)
    plt.close()
