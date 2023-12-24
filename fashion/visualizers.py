import matplotlib.pyplot as plt


def plot_losses(
    iteration_list,
    train_losses,
    test_losses,
    accuracy_scores,
    precision_scores,
    recall_scores,
    f1_scores,
):
    _, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].plot(iteration_list, train_losses)
    axs[0, 0].title.set_text("Train loss by iterations")

    axs[0, 1].plot(iteration_list, test_losses)
    axs[0, 1].title.set_text("Test loss by iterations")

    axs[0, 2].plot(iteration_list, accuracy_scores)
    axs[0, 2].title.set_text("Accuracy scores by iterations")

    axs[1, 0].plot(iteration_list, precision_scores)
    axs[1, 0].title.set_text("Precision scores by iterations")

    axs[1, 1].plot(iteration_list, recall_scores)
    axs[1, 1].title.set_text("Recall scores by iterations")

    axs[1, 2].plot(iteration_list, f1_scores)
    axs[1, 2].title.set_text("F1 scores by iterations")

    plt.show()
