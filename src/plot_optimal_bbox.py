import matplotlib.pyplot as plt


def main():
    # Data
    x_value = [26, 30, 34, 38, 42, 46, 50, 54, 58]  # bbox sizes
    f1_train = [0.848473421, 0.885654136, 0.873199409, 0.885158937, 0.885753198, 0.874609593, 0.900506079, 0.869609715, 0.863266852]
    f1_val = [0.845634211, 0.87767041, 0.868022729, 0.878959137, 0.877188585, 0.870006221, 0.877955534, 0.857403912, 0.862295229]
    pr_train = [0.896947878, 0.925876763, 0.917608235, 0.919313418, 0.898887701, 0.89702623, 0.905514507, 0.881999959, 0.887364778]
    pr_val = [0.89052919, 0.916948804, 0.910569498, 0.91223084, 0.891140726, 0.889470199, 0.886037153, 0.866905402, 0.887213067]
    rc_train = [0.809712138, 0.85224992, 0.837028454, 0.856909858, 0.876318292, 0.856955465, 0.897882534, 0.860294364, 0.843318545]
    rc_val = [0.810363673, 0.846011246, 0.834329451, 0.852670462, 0.867533261, 0.854864273, 0.873099305, 0.851047133, 0.841783243]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot training metrics
    ax1.plot(x_value, f1_train, label="f1_train", marker='o', markersize=8, c='tomato')
    ax1.plot(x_value, pr_train, label="pr_train", marker='o', markersize=8, c='mediumseagreen')
    ax1.plot(x_value, rc_train, label="rc_train", marker='o', markersize=8, c='royalblue')

    # Add annotations for training metrics
    for i, txt in enumerate(f1_train):
        ax1.annotate(f'{txt:.4f}', (x_value[i], f1_train[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, txt in enumerate(pr_train):
        ax1.annotate(f'{txt:.4f}', (x_value[i], pr_train[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, txt in enumerate(rc_train):
        ax1.annotate(f'{txt:.4f}', (x_value[i], rc_train[i]), textcoords="offset points", xytext=(0,10), ha='center')

    ax1.set_xlim(x_value[0]-2, x_value[-1]+2)
    ax1.set_xticks(range(x_value[0], x_value[-1]+1, 4))
    ax1.set_ylim(0.8, 1.0)
    ax1.set_yticks([])
    ax1.set_xlabel("Bounding box size")
    ax1.grid(True, linestyle="--")
    ax1.set_title("Training Metrics")
    ax1.legend()

    # Plot validation metrics
    ax2.plot(x_value, f1_val, label="f1_val", marker='o', markersize=8, c='tomato')
    ax2.plot(x_value, pr_val, label="pr_val", marker='o', markersize=8, c='mediumseagreen')
    ax2.plot(x_value, rc_val, label="rc_val", marker='o', markersize=8, c='royalblue')

    # Add annotations for validation metrics
    for i, txt in enumerate(f1_val):
        ax2.annotate(f'{txt:.4f}', (x_value[i], f1_val[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, txt in enumerate(pr_val):
        ax2.annotate(f'{txt:.4f}', (x_value[i], pr_val[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, txt in enumerate(rc_val):
        ax2.annotate(f'{txt:.4f}', (x_value[i], rc_val[i]), textcoords="offset points", xytext=(0,10), ha='center')

    ax2.set_xlim(x_value[0]-2, x_value[-1]+2)
    ax2.set_xticks(range(x_value[0], x_value[-1]+1, 4))
    ax2.set_ylim(0.8, 1.0)
    ax2.set_yticks([])
    ax2.set_xlabel("Bounding box size")
    ax2.grid(True, linestyle="--")
    ax2.set_title("Evaluation Metrics")
    ax2.legend()

    fig.suptitle("Finding Optimal Bounding Box Size")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    