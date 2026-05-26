import numpy as np
import matplotlib.pyplot as plt

def main():
    checkpoint = 29

    f_all = np.load(f"results/final_test/{checkpoint}/f.npy")   # shape (pop, 3)
    x_all = np.load(f"results/final_test/{checkpoint}/x.npy")   # shape (pop, n_params)

    # Extract objectives
    flat = f_all[:, 0]
    ice = f_all[:, 1]
    hill = f_all[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Flat vs Hill
    axes[0].scatter(flat, hill)
    axes[0].set_xlabel("Flat")
    axes[0].set_ylabel("Hill")
    axes[0].set_title("Flat vs Hill")

    # Flat vs Ice
    axes[1].scatter(flat, ice)
    axes[1].set_xlabel("Flat")
    axes[1].set_ylabel("Ice")
    axes[1].set_title("Flat vs Ice")

    # Ice vs Hill
    axes[2].scatter(ice, hill)
    axes[2].set_xlabel("Ice")
    axes[2].set_ylabel("Hill")
    axes[2].set_title("Ice vs Hill")

    plt.suptitle("Pareto Fronts")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()