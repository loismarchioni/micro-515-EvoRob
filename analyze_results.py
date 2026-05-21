import numpy as np
import matplotlib.pyplot as plt

def main():
    checkpoint = 49

    f_all = np.load(f"results/final_test/{checkpoint}/f.npy")   # shape (pop, 3)
    x_all = np.load(f"results/final_test/{checkpoint}/x.npy")   # shape (pop, n_params)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(f_all[:, 0], f_all[:, 1], f_all[:, 2])
    ax.set_xlabel("Flat"); ax.set_ylabel("Ice"); ax.set_zlabel("Hill")
    plt.show()

if __name__ == "__main__":
    main()