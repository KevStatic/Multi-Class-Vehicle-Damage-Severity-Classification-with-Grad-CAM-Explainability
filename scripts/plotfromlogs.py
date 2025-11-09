import matplotlib.pyplot as plt

# Paste values from your console output here:
train_acc_list = [92.79, 95.81, 96.68, 97.17, 97.60, 98.23, 98.65, 98.74, 98.91, 99.17]
val_acc_list = [92.79, 95.81, 96.68, 94.19, 98.62, 97.70, 98.62, 97.80, 98.60, 97.27]
loss_list = [0.1996, 0.1185, 0.0948, 0.0741, 0.0658, 0.0519, 0.0369, 0.0351, 0.0307, 0.0250]

epochs = range(1, 11)

plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc_list, label='Train Acc', marker='o')
plt.plot(epochs, val_acc_list, label='Val Acc', marker='o')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(epochs, loss_list, label='Training Loss', color='red', marker='x')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
