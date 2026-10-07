import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, labels):

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    return fig


def plot_metrics(results):

    metrics = {
        "Accuracy": results["accuracy"],
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1 Score": results["f1_score"]
    }

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(metrics.keys(), metrics.values())

    ax.set_ylim(0, 1)

    ax.set_title("Evaluation Metrics")

    return fig
