from pathlib import Path

import cv2
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "imagesDeTest" / "squares.tif"
OUTPUT_DIR = ROOT / "screen"


def rotate(image, angle, interpolation=cv2.INTER_NEAREST):
    """Rotate an image around its centre without changing its dimensions."""
    height, width = image.shape[:2]
    centre = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interpolation,
    )


def successive_rotations(
    image, angle, number_of_rotations, interpolation=cv2.INTER_NEAREST
):
    """Apply the same rotation repeatedly to the result of the previous one."""
    result = image.copy()
    for _ in range(number_of_rotations):
        result = rotate(result, angle, interpolation)
    return result


def main():
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {IMAGE_PATH}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    results = {
        "rotation-directe-90-horaire.png": rotate(image, -90),
        "rotation-2x45-horaire.png": successive_rotations(image, -45, 2),
        "rotation-6x15-horaire.png": successive_rotations(image, -15, 6),
        "rotation-directe-90-horaire-lineaire.png": rotate(
            image, -90, cv2.INTER_LINEAR
        ),
        "rotation-2x45-horaire-lineaire.png": successive_rotations(
            image, -45, 2, cv2.INTER_LINEAR
        ),
        "rotation-6x15-horaire-lineaire.png": successive_rotations(
            image, -15, 6, cv2.INTER_LINEAR
        ),
    }

    for filename, result in results.items():
        if not cv2.imwrite(str(OUTPUT_DIR / filename), result):
            raise OSError(f"Impossible d'enregistrer l'image : {filename}")

    comparison = [
        (
            "INTER_NEAREST",
            [
                results["rotation-directe-90-horaire.png"],
                results["rotation-2x45-horaire.png"],
                results["rotation-6x15-horaire.png"],
            ],
        ),
        (
            "INTER_LINEAR",
            [
                results["rotation-directe-90-horaire-lineaire.png"],
                results["rotation-2x45-horaire-lineaire.png"],
                results["rotation-6x15-horaire-lineaire.png"],
            ],
        ),
    ]
    titles = ["1 × 90° horaire", "2 × 45° horaires", "6 × 15° horaires"]
    figure, axes = plt.subplots(2, 3, figsize=(10, 7))
    for row, (interpolation, images) in enumerate(comparison):
        for column, (axis, title, result) in enumerate(
            zip(axes[row], titles, images)
        ):
            axis.imshow(result, cmap="gray", vmin=0, vmax=255)
            axis.set_title(title)
            axis.axis("off")
        axes[row, 0].set_ylabel(interpolation, fontsize=11)
    figure.suptitle("Comparaison des rotations : angle total de 90°")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "rotation-comparaison.png", dpi=150)
    plt.close(figure)

    print(f"Image originale : {image.shape[1]} x {image.shape[0]}")
    for filename, result in results.items():
        print(f"{filename} : {result.shape[1]} x {result.shape[0]}")


if __name__ == "__main__":
    main()
