import matplotlib.pyplot as plt

def graph_losses(train_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
#     plt.plot(test_losses, label='Test Loss')

    plt.title('Loss vs. Epoch', fontsize=14, y=1.03)
    plt.ylabel('Loss', fontsize=12, labelpad=10)
    plt.xlabel('Epoch', fontsize=12, labelpad=10)
    plt.legend(loc='upper right', bbox_to_anchor=(.99, .98))
    plt.show()