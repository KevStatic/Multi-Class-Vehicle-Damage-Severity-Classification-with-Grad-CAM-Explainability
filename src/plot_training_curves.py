import matplotlib.pyplot as plt
import os

def plot_curves(train_losses, val_losses, train_accs, val_accs):
    os.makedirs("results/plots", exist_ok=True)

    # Loss curve
    plt.figure(figsize=(6,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/plots/loss_curve.png", dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(6,5))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("results/plots/accuracy_curve.png", dpi=300)
    plt.close()

    print("\nPlots saved → results/plots/")
