import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# todo set params
out_path = "drive/MyDrive" #drive/MyDrive
test_name = "dga_domains"
class_names = ["legit", "dga"]
title="Bilbo"
normalize = True
cm = np.array([[32854.4, 885.4], [1115.9, 32634.1]]) #[[TN, FP], [FN, TP]]

def plot_confusion_matrix(cm, classes, normalize=normalize,
                          title=title,
                          cmap=plt.cm.Blues):
    fig = plt.figure(figsize=(4, 3), dpi=80)
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix:")
    else:
        print("\nConfusion matrix, without normalization:")
    print(cm)
    print("")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],5),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    fig.savefig(out_path + "/" + title + "_" + test_name + "_cm_averaged.png")
    plt.show()
    plt.close()

def main():
    plot_confusion_matrix(cm, class_names)
    print("Confusion matrix plotted")

if __name__ == "__main__":
    main()
