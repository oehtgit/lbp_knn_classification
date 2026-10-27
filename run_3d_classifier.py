from pathlib import Path
import argparse
import os
import re

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np

import lbp_knn_pipeline as lbp


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIEW_SIZE = 150
MODELNET_RENDER_NAME = re.compile(
    r"^(?P<class>.+?)_(?P<sample>\d+)"
    r"(?:\.obj\.shaded_v\d+|_\d+)"
    r"\.(?:png|jpe?g)$",
    re.IGNORECASE,
)


def list_samples(split_dir: Path):
    """Read either class/object/view folders or flat ModelNet rendered views."""
    samples = []

    if not split_dir.is_dir():
        return samples

    grouped_views = {}
    for image_path in sorted(split_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        match = MODELNET_RENDER_NAME.match(image_path.name)
        if match:
            class_name = match.group("class").lower()
            sample_id = match.group("sample")
            grouped_views.setdefault((class_name, sample_id), []).append(image_path)

    if grouped_views:
        return [
            (class_name, sample_id, image_paths)
            for (class_name, sample_id), image_paths in sorted(grouped_views.items())
            if len(image_paths) >= 6
        ]

    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for sample_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            image_paths = sorted(
                path
                for path in sample_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if image_paths:
                samples.append((class_dir.name, sample_dir.name, image_paths))

    return samples


def find_split_directory(dataset_root: Path, split: str, version: str = None):
    direct_dir = dataset_root / split
    if direct_dir.is_dir():
        return direct_dir

    def is_split_dir(path: Path):
        name = path.name.lower()
        modelnet_version = re.search(r"modelnet\d+v(\d+)", name)
        if version and (not modelnet_version or modelnet_version.group(1) != version):
            return False
        if split == "train":
            return "modelnet" in name and "train" in name and "shaded" in name
        return "modelnet" in name and "test" in name and "shaded" in name

    matches = []
    ignored_directories = {".git", "__pycache__", ".lbp_montages", "dataset2", "test1"}
    pending = [(dataset_root, 0)]
    while pending:
        current, depth = pending.pop()
        if not current.is_dir():
            continue
        if current != dataset_root and is_split_dir(current):
            matches.append(current)
            continue
        if depth >= 3:
            continue

        try:
            children = [
                child
                for child in current.iterdir()
                if child.is_dir() and child.name.lower() not in ignored_directories
            ]
        except PermissionError:
            continue
        pending.extend((child, depth + 1) for child in children)

    matches.sort(key=lambda path: (len(path.parts), str(path).lower()))
    if matches:
        return matches[0]
    return None


def get_modelnet_version(directory: Path):
    match = re.search(r"modelnet\d+v(\d+)", directory.name.lower())
    return match.group(1) if match else None


def resize_with_padding(image: np.ndarray, size: int = VIEW_SIZE):
    height, width = image.shape
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size), 127, dtype=np.uint8)
    x_offset = (size - resized_width) // 2
    y_offset = (size - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return canvas


def create_six_view_montage(image_paths, output_path: Path, rotation_transform=None):
    if len(image_paths) < 6:
        raise ValueError(
            f"Il faut au moins 6 images par objet ; {len(image_paths)} trouvée(s) dans "
            f"{image_paths[0].parent if image_paths else 'un dossier vide'}."
        )

    selected_indices = np.linspace(0, len(image_paths) - 1, num=6, dtype=int)
    views = []

    replay = None
    for index in selected_indices:
        image_path = image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image illisible : {image_path}")
        view = resize_with_padding(image)
        if rotation_transform is not None:
            if replay is None:
                augmented = rotation_transform(image=view)
                view = augmented["image"]
                replay = augmented["replay"]
            else:
                view = A.ReplayCompose.replay(replay, image=view)["image"]
        views.append(view)

    rows = [
        np.hstack(views[0:3]),
        np.hstack(views[3:6]),
    ]
    montage = np.vstack(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), montage):
        raise OSError(f"Impossible d'enregistrer la planche multi-vues : {output_path}")


def prepare_split(samples, split_name: str, cache_root: Path, class_to_label):
    entries = []
    split_cache = cache_root / split_name

    for class_name, sample_id, image_paths in samples:
        relative_path = Path(class_name) / f"{sample_id}.png"
        create_six_view_montage(
            image_paths, split_cache / relative_path, rotation_transform=None
        )
        entries.append((relative_path.as_posix(), class_to_label[class_name]))

    return entries, split_cache


def prepare_rotated_split(
    samples,
    split_name: str,
    cache_root: Path,
    class_to_label,
    augmenter,
    augmentations: int,
):
    entries = []
    split_cache = cache_root / f"{split_name}_rotated"

    for class_name, sample_id, image_paths in samples:
        for augmentation_index in range(augmentations):
            relative_path = Path(class_name) / (
                f"{sample_id}_rotation_{augmentation_index + 1:03d}.png"
            )
            create_six_view_montage(
                image_paths,
                split_cache / relative_path,
                rotation_transform=augmenter,
            )
            entries.append((relative_path.as_posix(), class_to_label[class_name]))

    return entries, split_cache


def predict_ssd(train_features, train_labels, test_features):
    predictions = []
    for test_sample in test_features:
        distances = np.abs(train_features - test_sample).sum(axis=1)
        predictions.append(int(train_labels[np.argmin(distances)]))
    return np.asarray(predictions, dtype=int)


def normalize_histogram(histogram):
    histogram = np.asarray(histogram, dtype=np.float32).reshape(1, -1)
    total = histogram.sum()
    if total > 0:
        histogram /= total
    else:
        histogram.fill(1.0 / histogram.size)
    return histogram


def predict_bhattacharyya(train_features, train_labels, test_features):
    normalized_train = [normalize_histogram(feature) for feature in train_features]
    predictions = []

    for test_sample in test_features:
        normalized_test = normalize_histogram(test_sample)
        distances = [
            cv2.compareHist(normalized_test, train_hist, cv2.HISTCMP_BHATTACHARYYA)
            for train_hist in normalized_train
        ]
        predictions.append(int(train_labels[int(np.argmin(distances))]))

    return np.asarray(predictions, dtype=int)


def calculate_metrics(predictions, labels, test_samples, class_names):
    correct = int(np.count_nonzero(predictions == labels))
    total = len(labels)
    accuracy = 100.0 * correct / total if total else 0.0
    misclassified = []
    for index, (prediction, expected) in enumerate(zip(predictions, labels)):
        if prediction != expected:
            class_name, sample_id, _ = test_samples[index]
            misclassified.append((class_name, sample_id, class_names[int(prediction)]))

    return accuracy, correct, total, misclassified


def display_misclassified(misclassified, cache_root: Path, split_name: str, max_show: int):
    selected = misclassified if max_show == 0 else misclassified[:max_show]
    if not selected:
        return

    tiles = []
    for true_class, sample_id, predicted_class in selected:
        montage_path = cache_root / split_name / true_class / f"{sample_id}.png"
        image = cv2.imread(str(montage_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Impossible de relire la planche générée : {montage_path}")
        cv2.putText(
            image,
            f"reel={true_class} predit={predicted_class}",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(image)

    columns = min(3, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    tile_height, tile_width = tiles[0].shape[:2]
    grid = np.full((rows * tile_height, columns * tile_width, 3), 240, dtype=np.uint8)

    for index, image in enumerate(tiles):
        y = (index // columns) * tile_height
        x = (index % columns) * tile_width
        grid[y:y + tile_height, x:x + tile_width] = image

    try:
        cv2.imshow("Planches multi-vues mal classees", grid)
        cv2.waitKey(0)
    except cv2.error as exc:
        raise RuntimeError(
            "L'affichage des erreurs nécessite une session graphique. "
            "Relance sans --show-misclassified pour garder les résultats texte."
        ) from exc
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reconnaissance multi-catégories par LBP multiscale à partir de six "
            "vues de chaque objet."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT,
        help=(
            "Dossier contenant les répertoires extraits shaded_modelnet40*_train et "
            "modelnet40*_test, ou un dataset3d/train et dataset3d/test."
        ),
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        help="Répertoire d'images d'entraînement ; détecté automatiquement par défaut.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Répertoire d'images de test ; détecté automatiquement par défaut.",
    )
    parser.add_argument(
        "--train-max",
        type=int,
        default=0,
        help="Nombre maximal d'objets d'entraînement par catégorie ; 0 = tous.",
    )
    parser.add_argument(
        "--test-max",
        type=int,
        default=0,
        help="Nombre maximal d'objets de test par catégorie ; 0 = tous.",
    )
    parser.add_argument(
        "--augmentations",
        type=int,
        default=1,
        help="Nombre de versions pivotées à ajouter par objet train ; 0 = aucune.",
    )
    parser.add_argument(
        "--show-misclassified",
        action="store_true",
        help="Affiche les planches des objets mal classés.",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=0,
        help="Nombre maximal de planches à afficher ; 0 = toutes.",
    )
    args = parser.parse_args()

    if (
        args.train_max < 0
        or args.test_max < 0
        or args.max_show < 0
        or args.augmentations < 0
    ):
        parser.error("Les limites doivent être positives ou égales à zéro.")

    dataset_root = args.dataset.resolve()
    train_dir = args.train_dir.resolve() if args.train_dir else find_split_directory(
        dataset_root, "train"
    )
    train_version = get_modelnet_version(train_dir) if train_dir else None
    test_dir = (
        args.test_dir.resolve()
        if args.test_dir
        else find_split_directory(dataset_root, "test", train_version)
    )

    if train_dir is None or test_dir is None:
        print(
            "Impossible de trouver les répertoires train/test.\n"
            "Décompresse les deux archives dans un même dossier, par exemple :\n"
            f"{dataset_root}\\shaded_modelnet40v1_train\\...\n"
            f"{dataset_root}\\shaded_modelnet40v1_test\\...\n"
            "Ou passe leurs chemins explicitement avec --train-dir et --test-dir."
        )
        return 2

    test_version = get_modelnet_version(test_dir)
    if train_version and test_version and train_version != test_version:
        raise ValueError(
            f"Les versions ModelNet ne correspondent pas : train v{train_version}, "
            f"test v{test_version}."
        )

    train_samples = list_samples(train_dir)
    test_samples = list_samples(test_dir)

    if not train_samples or not test_samples:
        print(
            "Aucun objet avec au moins 6 images n'a été trouvé.\n"
            f"Répertoire train : {train_dir}\n"
            f"Répertoire test : {test_dir}\n"
            "Vérifie que les archives sont extraites et que les images des vues "
            "sont présentes."
        )
        return 2

    train_counts = {}
    test_counts = {}
    for class_name, _, _ in train_samples:
        train_counts[class_name] = train_counts.get(class_name, 0) + 1
    for class_name, _, _ in test_samples:
        test_counts[class_name] = test_counts.get(class_name, 0) + 1

    class_names = sorted(set(train_counts) | set(test_counts))
    missing_train = sorted(set(test_counts) - set(train_counts))
    missing_test = sorted(set(train_counts) - set(test_counts))
    if missing_train or missing_test:
        raise ValueError(
            "Chaque catégorie doit avoir des objets en entraînement et en test. "
            f"Sans entraînement : {missing_train}; sans test : {missing_test}."
        )

    if args.train_max:
        limited_train = []
        counts = {}
        for sample in train_samples:
            class_name = sample[0]
            if counts.get(class_name, 0) < args.train_max:
                limited_train.append(sample)
                counts[class_name] = counts.get(class_name, 0) + 1
        train_samples = limited_train

    if args.test_max:
        limited_test = []
        counts = {}
        for sample in test_samples:
            class_name = sample[0]
            if counts.get(class_name, 0) < args.test_max:
                limited_test.append(sample)
                counts[class_name] = counts.get(class_name, 0) + 1
        test_samples = limited_test

    class_to_label = {class_name: index for index, class_name in enumerate(class_names)}
    cache_root = dataset_root / ".lbp_montages"
    train_entries, train_cache = prepare_split(
        train_samples, "train", cache_root, class_to_label
    )
    test_entries, test_cache = prepare_split(
        test_samples, "test", cache_root, class_to_label
    )
    rotation_transform = A.ReplayCompose(
        [
            A.Rotate(
                limit=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=127,
                p=1.0,
            )
        ]
    )

    lbp.IMAGE_MODE = "grayscale"
    lbp.PATCHES_DIR = train_cache
    baseline_train_descriptor_path = ROOT / "Training_dataset3d_multiscale.txt"
    lbp.build_dataset_multiscale(
        train_entries,
        baseline_train_descriptor_path,
        radii=(1, 2, 3),
        neighbors=8,
    )
    baseline_train_features, baseline_train_labels = lbp.load_dataset(
        baseline_train_descriptor_path
    )

    if args.augmentations:
        rotated_train_entries, rotated_train_cache = prepare_rotated_split(
            train_samples,
            "train",
            cache_root,
            class_to_label,
            rotation_transform,
            args.augmentations,
        )
        lbp.PATCHES_DIR = rotated_train_cache
        rotated_train_descriptor_path = (
            ROOT / "Training_dataset3d_multiscale_rotated.txt"
        )
        lbp.build_dataset_multiscale(
            rotated_train_entries,
            rotated_train_descriptor_path,
            radii=(1, 2, 3),
            neighbors=8,
        )
        rotated_train_features, rotated_train_labels = lbp.load_dataset(
            rotated_train_descriptor_path
        )
    else:
        rotated_train_features = np.empty(
            (0, baseline_train_features.shape[1]), dtype=baseline_train_features.dtype
        )
        rotated_train_labels = np.empty((0,), dtype=baseline_train_labels.dtype)

    lbp.PATCHES_DIR = test_cache
    test_descriptor_path = ROOT / "Test_dataset3d_multiscale.txt"
    lbp.build_dataset_multiscale(
        test_entries, test_descriptor_path, radii=(1, 2, 3), neighbors=8
    )
    test_features, test_labels = lbp.load_dataset(test_descriptor_path)

    test_original_count = len(test_samples)
    baseline_ssd_result = calculate_metrics(
        predict_ssd(
            baseline_train_features, baseline_train_labels, test_features
        ),
        test_labels,
        test_samples,
        class_names,
    )
    baseline_bhattacharyya_result = calculate_metrics(
        predict_bhattacharyya(
            baseline_train_features, baseline_train_labels, test_features
        ),
        test_labels,
        test_samples,
        class_names,
    )
    augmented_ssd_result = calculate_metrics(
        predict_ssd(
            np.vstack((baseline_train_features, rotated_train_features)),
            np.concatenate((baseline_train_labels, rotated_train_labels)),
            test_features,
        ),
        test_labels,
        test_samples,
        class_names,
    )
    augmented_bhattacharyya_result = calculate_metrics(
        predict_bhattacharyya(
            np.vstack((baseline_train_features, rotated_train_features)),
            np.concatenate((baseline_train_labels, rotated_train_labels)),
            test_features,
        ),
        test_labels,
        test_samples,
        class_names,
    )

    augmented_train_count = len(train_samples) + len(rotated_train_labels)
    print("\nComparaison LBP multiscale — test inchangé")
    print(
        f"Train : {len(train_samples)} originaux / "
        f"{augmented_train_count} avec rotation ±5° | "
        f"Test : {test_original_count}"
    )
    print(
        f"SSD            : sans augmentation "
        f"{baseline_ssd_result[0]:.2f}% "
        f"({baseline_ssd_result[1]}/{baseline_ssd_result[2]}) | "
        f"avec rotation {augmented_ssd_result[0]:.2f}% "
        f"({augmented_ssd_result[1]}/{augmented_ssd_result[2]})"
    )
    print(
        f"Bhattacharyya  : sans augmentation "
        f"{baseline_bhattacharyya_result[0]:.2f}% "
        f"({baseline_bhattacharyya_result[1]}/{baseline_bhattacharyya_result[2]}) | "
        f"avec rotation {augmented_bhattacharyya_result[0]:.2f}% "
        f"({augmented_bhattacharyya_result[1]}/{augmented_bhattacharyya_result[2]})"
    )

    if args.show_misclassified:
        display_misclassified(
            augmented_ssd_result[3], cache_root, "test", args.max_show
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
