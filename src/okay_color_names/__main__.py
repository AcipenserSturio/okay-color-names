import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from sklearn.cluster import KMeans
from PIL import Image
from .color import (
    is_valid_oklab,
    oklab_to_rgb,
    rgb_to_rgbfloat,
    hsl_to_rgb,
    rgb_to_oklab,
)


def rescale(value, low, high):
    return value*(high-low) + low


def scatter(lattice: list[tuple[float, float, float]]):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        *np.transpose(lattice),
        c=list(map(lambda x: rgb_to_rgbfloat(oklab_to_rgb(x)), lattice))
    )
    ax.set_xlabel('L (black-white)')
    ax.set_ylabel('a (green-red)')
    ax.set_zlabel('b (blue-yellow)')
    plt.show()


WIDTH = 360
HEIGHT = 100


def dist(a: list, b: list) -> float:
    return sum((x1-x2)**2 for x1, x2 in zip(a, b))


def hsl_img(centres: list[tuple[float, float, float]]):
    # # Test image size:
    # data = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # # Test HSL spectrum:
    # data = []
    # for l in range(0, HEIGHT):
    #     row = []
    #     for h in range(0, WIDTH):
    #         colour = hsl_to_rgb((h/WIDTH*360, 50, l/HEIGHT*100))
    #         row.append(colour)
    #     data.append(np.array(row, dtype=np.uint8))
    # data = np.array(data, dtype=np.uint8)
    # print(data)

    # Show closest centres:
    data = []
    for l in range(0, HEIGHT):
        row = []
        for h in range(0, WIDTH):
            colour = hsl_to_rgb((h/WIDTH*360, 50, l/HEIGHT*100))
            colour = rgb_to_oklab(colour)
            closest_centre = min(
                centres, key=lambda centre: dist(centre, colour)
            )
            row.append(oklab_to_rgb(closest_centre))
        data.append(np.array(row, dtype=np.uint8))
    data = np.array(data, dtype=np.uint8)

    img = Image.fromarray(data)
    img.show()


if __name__ == "__main__":
    lattice = np.random.uniform(0, 1, (50**3, 3))
    # This is essentially matrix multiplication
    # but i don't remember matrix multiplication so
    lattice = np.array([
        np.array([
            rescale(i, 0, 1),
            rescale(j, -.3, .3),
            rescale(k, -.3, .3),
        ])
        for i, j, k in lattice
    ])

    # TODO: instead of filtering away values that are outside RGB,
    # Consider readding clamping and then filtering out any oklab value
    # that gets moved too far after back-conversion
    # To deal with the black-point flared base
    with Pool(processes=6) as pool:
        mask = pool.map(is_valid_oklab, lattice)
    lattice = lattice[mask]

    # Visualise oklab space
    # scatter(lattice)

    COLORS = 100
    kmeans = KMeans(
        init="random",
        n_clusters=COLORS,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(lattice)
    # print(kmeans.cluster_centers_)
    # scatter(kmeans.cluster_centers_)

    hsl_img(kmeans.cluster_centers_)
