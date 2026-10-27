from pathlib import Path
import csv
import json
import time
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

ROOT = Path(__file__).resolve().parent
PATCHES_DIR = ROOT / 'PATCHES'
LABELS_DIR = ROOT / 'LABELS'

BENCHMARK_COMBOS = [
    (100, 100),
    (500, 500),
    (1000, 1000),
    (2000, 2000),
    (5000, 5000),
]


def read_entries(label_file: Path, limit=None):
    entries = []
    with label_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entries.append((parts[0], int(parts[1])))
            if limit is not None and len(entries) >= limit:
                break
    return entries


def compute_lbp_histogram(image_path: Path):
    gray = np.asarray(Image.open(image_path).convert('L'), dtype=np.uint8)
    h, w = gray.shape
    padded = np.pad(gray, ((1, 1), (1, 1)), mode='edge')
    center = padded[1:-1, 1:-1]
    offsets = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    lbp = np.zeros((h, w), dtype=np.uint16)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        lbp |= ((neighbor >= center).astype(np.uint16) << bit)
    hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float64)
    return hist


def build_dataset(entries, output_txt: Path):
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open('w', encoding='utf-8') as out:
        for rel_path, label in entries:
            image_path = PATCHES_DIR / rel_path
            if not image_path.exists():
                raise FileNotFoundError(f'Image not found: {image_path}')
            hist = compute_lbp_histogram(image_path)
            values = ' '.join(str(v) for v in hist.tolist())
            out.write(f'{values} {int(label)}\n')


def load_dataset(dataset_txt: Path):
    data = np.loadtxt(dataset_txt, delimiter=' ', dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    features = data[:, :-1]
    labels = data[:, -1].astype(int)
    return features, labels


def classify_test(train_features, train_labels, test_features, test_labels):
    correct = 0
    total = len(test_labels)
    for i, test_sample in enumerate(test_features):
        distances = np.abs(train_features - test_sample).sum(axis=1)
        predicted = int(train_labels[np.argmin(distances)])
        if predicted == int(test_labels[i]):
            correct += 1
    accuracy = (correct / total) * 100 if total else 0.0
    return accuracy, correct, total


def run_single_benchmark(train_size, test_size):
    train_entries = read_entries(LABELS_DIR / 'train.txt', limit=train_size)
    test_entries = read_entries(LABELS_DIR / 'test.txt', limit=test_size)

    if len(train_entries) < train_size or len(test_entries) < test_size:
        raise ValueError('Taille demandée supérieure aux fichiers disponibles')

    train_txt = ROOT / 'tmp_train.txt'
    test_txt = ROOT / 'tmp_test.txt'

    t0 = time.perf_counter()
    build_dataset(train_entries, train_txt)
    train_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    build_dataset(test_entries, test_txt)
    test_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    train_features, train_labels = load_dataset(train_txt)
    test_features, test_labels = load_dataset(test_txt)
    load = time.perf_counter() - t0

    t0 = time.perf_counter()
    accuracy, correct, total = classify_test(train_features, train_labels, test_features, test_labels)
    classify = time.perf_counter() - t0

    total_time = train_build + test_build + load + classify

    return {
        'train_size': train_size,
        'test_size': test_size,
        'train_build_s': train_build,
        'test_build_s': test_build,
        'load_s': load,
        'classify_s': classify,
        'total_s': total_time,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
    }


def save_results_csv(rows, output_csv):
    fieldnames = ['train_size', 'test_size', 'train_build_s', 'test_build_s', 'load_s', 'classify_s', 'total_s', 'accuracy', 'correct', 'total']
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report_pdf(rows, csv_path, pdf_path):
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph('Rapport de benchmark LBP - temps et précision', styles['Title']))
    story.append(Spacer(1, 18))
    story.append(Paragraph('Ce rapport compare plusieurs tailles de jeux de données pour la méthode LBP + plus proche voisin.', styles['BodyText']))
    story.append(Spacer(1, 12))

    data = [['Train', 'Test', 'Temps total (s)', 'Précision (%)', 'Correct/Total']]
    for row in rows:
        data.append([
            str(row['train_size']),
            str(row['test_size']),
            f"{row['total_s']:.3f}",
            f"{row['accuracy']:.2f}",
            f"{row['correct']}/{row['total']}",
        ])

    table = Table(data, colWidths=[90, 90, 110, 100, 120])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C4E7A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D9EAF7')),
    ]))
    story.append(table)
    story.append(Spacer(1, 18))

    story.append(Paragraph(f'Fichier CSV généré : {csv_path.name}', styles['BodyText']))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    doc.build(story)


def main():
    rows = []
    for train_size, test_size in BENCHMARK_COMBOS:
        try:
            row = run_single_benchmark(train_size, test_size)
            rows.append(row)
            print(f'Train={train_size}, Test={test_size} -> {row["accuracy"]:.2f}% | {row["total_s"]:.3f}s')
        except Exception as exc:
            print(f'Skip Train={train_size}, Test={test_size}: {exc}')

    rows.sort(key=lambda r: (r['train_size'], r['test_size']))
    csv_path = ROOT / 'benchmark_results.csv'
    pdf_path = ROOT / 'benchmark_results.pdf'
    save_results_csv(rows, csv_path)
    build_report_pdf(rows, csv_path, pdf_path)

    print(f'CSV sauvegardé : {csv_path}')
    print(f'PDF sauvegardé : {pdf_path}')


if __name__ == '__main__':
    main()
