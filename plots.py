import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os




def loss_curve(train_loss, val_loss, plots_path):
    os.makedirs(plots_path, exist_ok=True)
    epoch_arr = [epoch + 1 for epoch in range(len(train_loss))]
    plt.plot(epoch_arr, train_loss, label='train loss')
    plt.plot(epoch_arr, val_loss, label='validation loss')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(plots_path, 'loss_curve.png'))





if __name__ == '__main__':

    train_loss = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    val_loss = [0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.55, 0.6, 0.6]
    loss_curve(train_loss, val_loss, '/Users/abhiramkandiyana/LLMsFromScratch/ViT/results/plots')

