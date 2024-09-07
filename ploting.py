import re
import matplotlib.pyplot as plt

# Define a function to extract data from the log file
def extract_data(filename):
    train_epochs = []
    val_epochs = []
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Match training data
            train_match = re.search(r'train E(\d+):.*acc=([0-9.]+), loss=([0-9.]+)[.\]]', line)
            if train_match:
                epoch = int(train_match.group(1))
                acc = float(train_match.group(2))
                loss = float(train_match.group(3))
                train_epochs.append(epoch)
                train_acc.append(acc)
                train_loss.append(loss)

            # Match validation data
            val_match = re.search(r'val E(\d+):.*acc=([0-9.]+), loss=([0-9.]+)[.\]]', line)
            if val_match:
                epoch = int(val_match.group(1))
                acc = float(val_match.group(2))
                loss = float(val_match.group(3))
                val_epochs.append(epoch)
                val_acc.append(acc)
                val_loss.append(loss)

    return train_epochs, train_acc, train_loss, val_epochs, val_acc, val_loss

# Define a function to plot the data
def plot_data(train_epochs, train_acc, train_loss, val_epochs, val_acc, val_loss):
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(train_epochs, train_acc, label='Train Accuracy')
    print("Acc: ")
    print(train_acc)
    plt.plot(val_epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_epochs, train_loss, label='Train Loss')
    plt.plot(val_epochs, val_loss, label='Validation Loss')
    print("Loss: ")
    print(val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

    print("Plots saved as 'accuracy_plot.png' and 'loss_plot.png'.")

# Main function
def main():
    filename = 'epoch.txt'  # Replace with your filename
    train_epochs, train_acc, train_loss, val_epochs, val_acc, val_loss = extract_data(filename)
    plot_data(train_epochs, train_acc, train_loss, val_epochs, val_acc, val_loss)

if __name__ == '__main__':
    main()
