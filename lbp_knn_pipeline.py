
from pathlib import Path

import argparse
import time
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
PATCHES_DIR = ROOT / "PATCHES"
LABELS_DIR = ROOT / "LABELS"

# Mode actuel :
# "grayscale" = fonctionnement original
# "packed_rgb" = image unique R | G | B
IMAGE_MODE = "packed_rgb"


def read_entries(label_file: Path, limit=None):
    """Read image paths and labels from LABELS/*.txt."""
    entries = []

    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            rel_path = parts[0]
            label = int(parts[1])

            entries.append((rel_path, label))

            if limit is not None and len(entries) >= limit:
                break

    return entries


def load_image(image_path: Path):
    """
    Charge une image selon le mode choisi.

    grayscale :
        Image originale convertie en niveaux de gris.
        Taille : H x W

    packed_rgb :
        Une seule image CV_8UC1 contenant :
        [ ROUGE | VERT | BLEU ]

        Si l'image originale fait H x W x 3,
        l'image résultante fait H x (3W).
    """

    if IMAGE_MODE == "grayscale":

        # Fonctionnement original
        image = np.asarray(
            Image.open(image_path).convert("L"),
            dtype=np.uint8
        )

        return image

    elif IMAGE_MODE == "packed_rgb":

        # PIL utilise RGB directement
        image = np.asarray(
            Image.open(image_path).convert("RGB"),
            dtype=np.uint8
        )

        # Séparation des trois composantes
        red = image[:, :, 0]
        green = image[:, :, 1]
        blue = image[:, :, 2]

        # Une seule image :
        # [ ROUGE | VERT | BLEU ]
        packed_image = np.hstack((red, green, blue))

        return packed_image

    else:
        raise ValueError(
            f"Mode d'image inconnu : {IMAGE_MODE}"
        )


def compute_lbp_histogram(image_path: Path, radius=1, neighbors=8):
    """
    Compute a 256-bin LBP histogram.

    En mode grayscale :
        image = H x W

    En mode packed_rgb :
        image = H x (3W)

    Dans les deux cas, le calcul LBP reste identique.
    La version actuelle reste inchangée en mode standard.
    """

    gray = load_image(image_path)
    return _compute_lbp_histogram_array(gray, radius=radius, neighbors=neighbors)


def _compute_lbp_histogram_array(gray, radius=1, neighbors=8):
    h, w = gray.shape

    padded = np.pad(
        gray,
        ((radius + 1, radius + 1), (radius + 1, radius + 1)),
        mode="edge"
    )

    center = padded[radius + 1:-radius - 1, radius + 1:-radius - 1]

    lbp = np.zeros((h, w), dtype=np.uint16)

    for bit in range(neighbors):
        angle = 2 * np.pi * bit / neighbors
        dx = int(round(radius * np.cos(angle)))
        dy = int(round(radius * np.sin(angle)))

        neighbor = padded[
            radius + 1 + dy : radius + 1 + dy + h,
            radius + 1 + dx : radius + 1 + dx + w
        ]

        lbp |= ((neighbor >= center).astype(np.uint16) << bit)

    hist = np.bincount(lbp.ravel(), minlength=256).astype(np.int64)
    return hist


def compute_multiscale_lbp_histogram(image_path: Path, radii=(1, 2, 3), neighbors=8):
    """
    Version conforme à la figure :
    - niveau 1 : 9 sous-régions
    - niveau 2 : 4 sous-régions
    - niveau 3 : 1 sous-région
    Total : 14 histogrammes LBP concaténés.
    """
    image = load_image(image_path)
    hist_parts = []

    for rows, cols in [(3, 3), (2, 2), (1, 1)]:
        h, w = image.shape
        y_step = h / rows
        x_step = w / cols

        for r in range(rows):
            for c in range(cols):
                y0 = int(r * y_step)
                y1 = int((r + 1) * y_step) if r < rows - 1 else h
                x0 = int(c * x_step)
                x1 = int((c + 1) * x_step) if c < cols - 1 else w

                block = image[y0:y1, x0:x1]
                hist = _compute_lbp_histogram_array(block, radius=1, neighbors=neighbors)
                hist_parts.append(hist)

    return np.concatenate(hist_parts).astype(np.float64)


def build_dataset(entries, output_txt: Path):
    """Create Training.txt / Test.txt with 256 descriptor values plus label."""

    output_txt.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_txt.open("w", encoding="utf-8") as out:

        for rel_path, label in entries:

            image_path = PATCHES_DIR / rel_path

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            hist = compute_lbp_histogram(image_path)

            values = " ".join(
                str(v)
                for v in hist.tolist()
            )

            out.write(
                f"{values} {int(label)}\n"
            )


def build_dataset_multiscale(entries, output_txt: Path, radii=(1, 2, 3), neighbors=8):
    """Version conforme à la figure : 9 + 4 + 1 sous-régions = 14 histogrammes concaténés."""

    output_txt.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_txt.open("w", encoding="utf-8") as out:
        for rel_path, label in entries:
            image_path = PATCHES_DIR / rel_path

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            hist = compute_multiscale_lbp_histogram(
                image_path,
                radii=radii,
                neighbors=neighbors,
            )

            values = " ".join(
                str(v)
                for v in hist.tolist()
            )

            out.write(
                f"{values} {int(label)}\n"
            )


def load_dataset(dataset_txt: Path):

    data = np.loadtxt(
        dataset_txt,
        delimiter=" ",
        dtype=np.float64
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    features = data[:, :-1]

    labels = data[:, -1].astype(int)

    return features, labels


def classify_test(
    train_features,
    train_labels,
    test_features,
    test_labels
):

    correct = 0

    total = len(test_labels)

    for i, test_sample in enumerate(test_features):

        distances = np.abs(
            train_features - test_sample
        ).sum(axis=1)

        predicted_label = int(
            train_labels[np.argmin(distances)]
        )

        if predicted_label == int(test_labels[i]):
            correct += 1

    accuracy = (
        (correct / total) * 100
        if total
        else 0.0
    )

    return accuracy, correct, total


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate LBP descriptors for "
            "training/test sets and compute "
            "recognition accuracy with "
            "nearest-neighbor classification."
        )
    )

    parser.add_argument(
        "--train-max",
        type=int,
        default=5000,
        help="Number of training images to process. Use 0 for all images.",
    )

    parser.add_argument(
        "--test-max",
        type=int,
        default=5000,
        help="Number of test images to process. Use 0 for all images.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Folder where Training.txt and Test.txt are written.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_entries = read_entries(
        LABELS_DIR / "train.txt",
        limit=None
        if args.train_max == 0
        else args.train_max
    )

    test_entries = read_entries(
        LABELS_DIR / "test.txt",
        limit=None
        if args.test_max == 0
        else args.test_max
    )

    train_txt = output_dir / "Training.txt"

    test_txt = output_dir / "Test.txt"

    start_total = time.perf_counter()

    train_build_start = time.perf_counter()

    build_dataset(
        train_entries,
        train_txt
    )

    train_build_duration = (
        time.perf_counter()
        - train_build_start
    )

    test_build_start = time.perf_counter()

    build_dataset(
        test_entries,
        test_txt
    )

    test_build_duration = (
        time.perf_counter()
        - test_build_start
    )

    load_start = time.perf_counter()

    train_features, train_labels = load_dataset(
        train_txt
    )

    test_features, test_labels = load_dataset(
        test_txt
    )

    load_duration = (
        time.perf_counter()
        - load_start
    )

    classify_start = time.perf_counter()

    accuracy, correct, total = classify_test(
        train_features,
        train_labels,
        test_features,
        test_labels
    )

    classification_duration = (
        time.perf_counter()
        - classify_start
    )

    total_duration = (
        time.perf_counter()
        - start_total
    )

    print(
        f"Mode image : {IMAGE_MODE}"
    )

    print(
        f"Nombre d'images training utilisées : "
        f"{len(train_entries)}"
    )

    print(
        f"Nombre d'images test utilisées : "
        f"{len(test_entries)}"
    )

    print(
        f"Temps d'apprentissage "
        f"(génération + chargement) : "
        f"{train_build_duration + load_duration:.3f} s"
    )

    print(
        f"Temps de traitement / classification : "
        f"{classification_duration:.3f} s"
    )

    print(
        f"Temps total : "
        f"{total_duration:.3f} s"
    )

    print(
        f"Précision du modèle : "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )

    print(
        f"Fichiers générés : "
        f"{train_txt}, {test_txt}"
    )

