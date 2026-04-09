import json
import pathlib
from typing import Dict, List, Optional

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def _load_epoch_records(log_path: str) -> List[Dict]:
    latest_by_epoch: Dict[int, Dict] = dict()

    with open(log_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = record.get('epoch')
            if epoch is None:
                continue

            try:
                epoch_index = int(epoch)
            except (TypeError, ValueError):
                continue

            latest_by_epoch[epoch_index] = record

    return [latest_by_epoch[epoch] for epoch in sorted(latest_by_epoch.keys())]


def plot_loss_curve(log_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """Create an English-language loss curve image from the training log."""
    if output_path is None:
        output_path = str(pathlib.Path(log_path).with_name('loss_curve_en.png'))

    epoch_records = _load_epoch_records(log_path)
    if len(epoch_records) == 0:
        return None

    epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    for record in epoch_records:
        if 'train_loss' not in record:
            continue

        epochs.append(int(record['epoch']))
        train_losses.append(float(record['train_loss']))
        val_losses.append(float(record['val_loss']) if 'val_loss' in record else float('nan'))

    if len(epochs) == 0:
        return None

    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    ax.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2.0)
    if any(not (value != value) for value in val_losses):
        ax.plot(epochs, val_losses, label='Validation Loss', color='#d62728', linewidth=2.0)

    ax.set_title('Training Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

    return str(output_file)