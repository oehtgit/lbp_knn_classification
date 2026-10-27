from pathlib import Path
import argparse
import time

import cv2
import numpy as np

import lbp_knn_pipeline as m

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'dataset2'


def ensure_label_files(dataset_dir: Path):
    train_txt = dataset_dir / 'train.txt'
    test_txt = dataset_dir / 'test.txt'

    if train_txt.exists() and test_txt.exists():
        return  

    train_lines = []
    test_lines = []

    for split_name, split_dir in [('train', dataset_dir / 'train'), ('test', dataset_dir / 'test')]:
        for cls_name in ['empty', 'occupied']:
            target_dir = split_dir / cls_name
            for img in sorted(target_dir.rglob('*.jpg')):
                rel = img.relative_to(dataset_dir).as_posix()
                label = 0 if cls_name == 'empty' else 1
                if split_name == 'train':
                    train_lines.append(f'{rel} {label}')
                else:
                    test_lines.append(f'{rel} {label}')

    train_txt.write_text('\n'.join(train_lines) + '\n', encoding='utf-8')
    test_txt.write_text('\n'.join(test_lines) + '\n', encoding='utf-8')


def classify_with_ssd(train_features, train_labels, test_features, test_labels):
    correct = 0
    total = len(test_labels)

    for i, test_sample in enumerate(test_features):
        distances = np.abs(train_features - test_sample).sum(axis=1)
        predicted_label = int(train_labels[np.argmin(distances)])
        if predicted_label == int(test_labels[i]):
            correct += 1

    accuracy = (correct / total) * 100 if total else 0.0
    return accuracy, correct, total


def normalize_hist(hist):
    hist = np.asarray(hist, dtype=np.float32).reshape(1, -1)
    hist_sum = hist.sum()
    if hist_sum <= 0:
        hist = np.full_like(hist, 1.0 / hist.size, dtype=np.float32)
    else:
        hist = hist / hist_sum
    return hist


def classify_with_bhattacharyya(train_features, train_labels, test_features, test_labels):
    normalized_train = [normalize_hist(train_feature) for train_feature in train_features]

    correct = 0
    total = len(test_labels)

    for i, test_sample in enumerate(test_features):
        test_hist = normalize_hist(test_sample)
        best_distance = None
        best_label = None

        for j, train_hist in enumerate(normalized_train):
            distance = cv2.compareHist(test_hist, train_hist, cv2.HISTCMP_BHATTACHARYYA)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_label = int(train_labels[j])

        if best_label == int(test_labels[i]):
            correct += 1

    accuracy = (correct / total) * 100 if total else 0.0
    return accuracy, correct, total


def find_misclassified_samples(train_features, train_labels, test_features, test_labels, test_entries, max_examples=0):
    misclassified = []

    for idx, test_sample in enumerate(test_features):
        distances = np.abs(train_features - test_sample).sum(axis=1)
        predicted_label = int(train_labels[np.argmin(distances)])
        true_label = int(test_labels[idx])

        if predicted_label != true_label:
            rel_path, _ = test_entries[idx]
            misclassified.append({
                'index': idx,
                'path': str(DATASET / rel_path),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'nearest_distance': float(np.min(distances)),
            })

            if max_examples > 0 and len(misclassified) >= max_examples:
                break

    return misclassified


def display_misclassified_images(misclassified_samples, max_per_row=5):
    if not misclassified_samples:
        print("Aucune image mal classée.")
        return

    print("\nImages mal classées :")

    try:
        images = []
        for item in misclassified_samples:
            image = cv2.imread(item['path'], cv2.IMREAD_COLOR)
            if image is None:
                image = cv2.imread(item['path'], cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image is not None:
                pad = 20
                text = f"true={item['true_label']} pred={item['predicted_label']}"
                color = (0, 0, 255) if item['true_label'] != item['predicted_label'] else (0, 255, 0)
                cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                images.append(image)

        if not images:
            print("Aucune image exploitable pour l’affichage visuel.")
            return

        n = len(images)
        cols = min(max_per_row, n)
        rows = (n + cols - 1) // cols

        h, w = images[0].shape[:2]
        grid_h = rows * h
        grid_w = cols * w
        grid = np.zeros((grid_h + 20, grid_w, 3), dtype=np.uint8)
        grid[:] = 240

        for idx, img in enumerate(images):
            r = idx // cols
            c = idx % cols
            y0 = r * h
            x0 = c * w
            grid[y0:y0 + h, x0:x0 + w] = img

        cv2.imshow("Misclassified images", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("Affichage visuel ignoré (pas de fenêtre graphique disponible).")


def run_single_mode(mode_name, train_entries, test_entries, train_txt, test_txt):
    m.PATCHES_DIR = DATASET
    m.LABELS_DIR = DATASET
    m.IMAGE_MODE = mode_name if mode_name in ['grayscale', 'packed_rgb'] else 'grayscale'

    start_total = time.perf_counter()

    train_build_start = time.perf_counter()
    if mode_name == 'multiscale':
        m.build_dataset_multiscale(train_entries, train_txt, radii=(1, 2, 3), neighbors=8)
    else:
        m.build_dataset(train_entries, train_txt)
    train_build_duration = time.perf_counter() - train_build_start

    test_build_start = time.perf_counter()
    if mode_name == 'multiscale':
        m.build_dataset_multiscale(test_entries, test_txt, radii=(1, 2, 3), neighbors=8)
    else:
        m.build_dataset(test_entries, test_txt)
    test_build_duration = time.perf_counter() - test_build_start

    load_start = time.perf_counter()
    train_features, train_labels = m.load_dataset(train_txt)
    test_features, test_labels = m.load_dataset(test_txt)
    load_duration = time.perf_counter() - load_start

    ssd_start = time.perf_counter()
    ssd_accuracy, ssd_correct, total = classify_with_ssd(train_features, train_labels, test_features, test_labels)
    ssd_duration = time.perf_counter() - ssd_start

    bh_start = time.perf_counter()
    bh_accuracy, bh_correct, total_bh = classify_with_bhattacharyya(train_features, train_labels, test_features, test_labels)
    bh_duration = time.perf_counter() - bh_start

    total_duration = time.perf_counter() - start_total

    return {
        'mode': mode_name,
        'train_count': len(train_entries),
        'test_count': len(test_entries),
        'learning_time': train_build_duration + load_duration,
        'ssd_time': ssd_duration,
        'bh_time': bh_duration,
        'ssd_accuracy': ssd_accuracy,
        'bh_accuracy': bh_accuracy,
        'ssd_correct': ssd_correct,
        'bh_correct': bh_correct,
        'total': total,
        'total_duration': total_duration,
        'train_txt': train_txt,
        'test_txt': test_txt,
    }


def print_results_table(results):
    headers = ['Mode', 'Train', 'Test', 'Learn(s)', 'SSD(s)', 'Bh(s)', 'SSD%', 'Bh%', 'Δ%']
    rows = []

    for item in results:
        rows.append([
            item['mode'],
            str(item['train_count']),
            str(item['test_count']),
            f"{item['learning_time']:.3f}",
            f"{item['ssd_time']:.3f}",
            f"{item['bh_time']:.3f}",
            f"{item['ssd_accuracy']:.2f}",
            f"{item['bh_accuracy']:.2f}",
            f"{abs(item['ssd_accuracy'] - item['bh_accuracy']):.3f}",
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def fmt_row(r):
        return ' | '.join(value.ljust(widths[i]) for i, value in enumerate(r))

    print('\nComparaison des modes de traitement')
    print(fmt_row(headers))
    print('-+-'.join('-' * width for width in widths))
    for row in rows:
        print(fmt_row(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-max', type=int, default=1000)
    parser.add_argument('--test-max', type=int, default=1000)
    parser.add_argument('--mode', choices=['grayscale', 'packed_rgb', 'multiscale', 'compare'], default='compare', help='grayscale = noir et blanc, packed_rgb = couleur empilée, multiscale = approche multi-échelle, compare = les trois')
    parser.add_argument('--show-misclassified', action='store_true', help='Affiche les images mal classées et leur chemin')
    parser.add_argument('--max-show', type=int, default=0, help='Nombre d’exemples mal classés à afficher. 0 = toutes les afficher.')
    args = parser.parse_args()

    ensure_label_files(DATASET)

    train_entries = m.read_entries(DATASET / 'train.txt', limit=None if args.train_max == 0 else args.train_max)
    test_entries = m.read_entries(DATASET / 'test.txt', limit=None if args.test_max == 0 else args.test_max)

    modes_to_run = [args.mode] if args.mode != 'compare' else ['grayscale', 'packed_rgb', 'multiscale']
    results = []

    for mode_name in modes_to_run:
        train_txt = ROOT / f'Training_dataset2_{mode_name}.txt'
        test_txt = ROOT / f'Test_dataset2_{mode_name}.txt'

        result = run_single_mode(mode_name, train_entries, test_entries, train_txt, test_txt)
        results.append(result)

        print(f"\nMode : {mode_name}")
        print(f"Nombre d'images training utilisées : {len(train_entries)}")
        print(f"Nombre d'images test utilisées : {len(test_entries)}")
        print(f"Temps d'apprentissage (génération + chargement) : {result['learning_time']:.3f} s")
        print(f"Temps de traitement / classification SSD : {result['ssd_time']:.3f} s")
        print(f"Temps de traitement / classification Bhattacharyya : {result['bh_time']:.3f} s")
        print(f"Temps total : {result['total_duration']:.3f} s")
        print(f"Précision SSD : {result['ssd_accuracy']:.2f}% ({result['ssd_correct']}/{result['total']})")
        print(f"Précision Bhattacharyya : {result['bh_accuracy']:.2f}% ({result['bh_correct']}/{result['total']})")
        print(f"Fichiers générés : {train_txt}, {test_txt}")

        if args.show_misclassified:
            train_features, train_labels = m.load_dataset(train_txt)
            test_features, test_labels = m.load_dataset(test_txt)
            misclassified = find_misclassified_samples(
                train_features,
                train_labels,
                test_features,
                test_labels,
                test_entries,
                max_examples=args.max_show,
            )
            display_misclassified_images(misclassified, max_per_row=5)

    if len(results) > 1:
        print_results_table(results)


def convert_rgb_to_packed_image(image_path):
    """
    Lit une image couleur et crée une seule image en niveaux de gris
    dont la largeur est multipliée par 3 :

    [ ROUGE | VERT | BLEU ]

    Retourne une image CV_8UC1 de taille H x (3W).
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    # OpenCV lit les images en BGR
    blue, green, red = cv2.split(image)

    # Une seule image, avec les 3 composantes côte à côte
    packed_image = np.hstack((red, green, blue))

    return packed_image 

if __name__ == '__main__':
    main()
