import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt




def loss_curve(train_loss, val_loss):
    epoch_arr = [epoch + 1 for epoch in range(len(train_loss))]
    plt.plot(epoch_arr, train_loss, label='train loss')
    plt.plot(epoch_arr, val_loss, label='validation loss')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()






if __name__ == '__main__':

    train_loss = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    val_loss = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.55, 0.6, 0.6]
    loss_curve(train_loss, val_loss)

