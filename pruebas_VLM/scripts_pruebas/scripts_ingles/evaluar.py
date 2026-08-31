"""
evaluar_predicciones.py
=======================
Calcula métricas de clasificación a partir de un CSV con columnas:
  - label              : etiqueta real
  - predicted_category : etiqueta predicha  (puede contener 'unknown')
  - confidence         : probabilidad/confianza del modelo (opcional, para curva ROC)

Manejo de 'unknown':
  - Las predicciones 'unknown' se tratan como errores (nunca cuentan como acierto).
  - Se reportan por separado: cuántas hay, qué clases reales representan,
    y cómo afectan a las métricas con dos estrategias:
      · EXCLUIDAS  → se eliminan del cálculo (métricas sobre lo que sí se predijo)
      · INCLUIDAS  → se tratan como clase extra (penalizan accuracy y F1)
  - La matriz de confusión muestra una columna adicional para 'unknown'.

Uso:
    python evaluar_predicciones.py predictions_fire_val.csv
    python evaluar_predicciones.py mi_archivo.csv --label label --pred predicted_category --conf confidence
    python evaluar_predicciones.py mi_archivo.csv --unknown desconocido   # nombre alternativo
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    f1_score,
)
from sklearn.preprocessing import label_binarize

# ──────────────────────────────────────────────
# CONFIGURACIÓN POR DEFECTO
# ──────────────────────────────────────────────
DEFAULT_CSV     = "predictions_fire_val.csv"
DEFAULT_LABEL   = "label"
DEFAULT_PRED    = "predicted_category"
DEFAULT_CONF    = "confidence"
DEFAULT_UNKNOWN = "unknown"
OUTPUT_PREFIX   = "metricas"


# ──────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────
def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_data(path, label_col, pred_col, conf_col):
    df = pd.read_csv(path)
    missing = [c for c in [label_col, pred_col] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Columnas no encontradas: {missing}\n"
                 f"   Disponibles: {df.columns.tolist()}")
    y_true = df[label_col].astype(str)
    y_pred = df[pred_col].astype(str)
    y_conf = df[conf_col].values if (conf_col and conf_col in df.columns) else None
    return y_true, y_pred, y_conf, df


# ──────────────────────────────────────────────
# ANÁLISIS DE UNKNOWN
# ──────────────────────────────────────────────
def analyze_unknown(y_true, y_pred, y_conf, unknown_label):
    mask_unk = y_pred == unknown_label
    n_unk    = mask_unk.sum()
    n_total  = len(y_pred)

    print_section(f"ANALISIS DE PREDICCIONES '{unknown_label.upper()}'")

    if n_unk == 0:
        print(f"  OK: No hay predicciones '{unknown_label}' en este CSV.")
        return y_true, y_pred, y_conf, mask_unk

    pct = n_unk / n_total * 100
    print(f"  Total muestras        : {n_total:,}")
    print(f"  Predicciones unknown  : {n_unk:,}  ({pct:.2f}%)")

    dist = y_true[mask_unk].value_counts()
    print(f"\n  Clase real de las '{unknown_label}' predichas:")
    for cls, cnt in dist.items():
        bar = "#" * int(cnt / dist.max() * 20)
        print(f"    {cls:<10}  {cnt:>5}  {bar}")

    if y_conf is not None:
        conf_unk = y_conf[mask_unk.values]
        print(f"\n  Confianza en unknown  -> media={conf_unk.mean():.3f}  "
              f"std={conf_unk.std():.3f}  "
              f"min={conf_unk.min():.3f}  max={conf_unk.max():.3f}")

    mask_known = ~mask_unk
    y_true_k   = y_true[mask_known].reset_index(drop=True)
    y_pred_k   = y_pred[mask_known].reset_index(drop=True)
    y_conf_k   = y_conf[mask_known.values] if y_conf is not None else None

    print(f"\n  Muestras tras excluir unknown: {mask_known.sum():,}")
    return y_true_k, y_pred_k, y_conf_k, mask_unk


def plot_unknown_breakdown(y_true, mask_unk, known_classes, unknown_label, output_prefix):
    if mask_unk.sum() == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    dist = y_true[mask_unk].value_counts().reindex(known_classes, fill_value=0)
    axes[0].bar(dist.index, dist.values, color="#E07B54")
    axes[0].set_title(f"Clase real de predicciones '{unknown_label}'")
    axes[0].set_xlabel("Clase real")
    axes[0].set_ylabel(f"N predicciones {unknown_label}")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(dist.values):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontsize=9)

    total_per_class = y_true.value_counts().reindex(known_classes, fill_value=1)
    rate = (dist / total_per_class * 100).fillna(0)
    axes[1].bar(rate.index, rate.values, color="#9B59B6")
    axes[1].set_title(f"Tasa de '{unknown_label}' por clase real (%)")
    axes[1].set_xlabel("Clase real")
    axes[1].set_ylabel("% predichos como unknown")
    axes[1].set_ylim(0, min(rate.max() * 1.25 + 1, 100))
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(rate.values):
        axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    path = f"{output_prefix}_unknown_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Guardada: {path}")
    plt.close()


# ──────────────────────────────────────────────
# METRICAS
# ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred, classes, label=""):
    title = f"METRICAS GLOBALES{' -- ' + label if label else ''}"
    print_section(title)

    acc         = accuracy_score(y_true, y_pred)
    bal_acc     = balanced_accuracy_score(y_true, y_pred)
    kappa       = cohen_kappa_score(y_true, y_pred)
    mcc         = matthews_corrcoef(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0, labels=classes)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=classes)
    err         = 1 - acc

    metrics = {
        "Accuracy":              acc,
        "Error Rate (ERR)":      err,
        "Balanced Accuracy":     bal_acc,
        "F1 Macro":              f1_macro,
        "F1 Weighted":           f1_weighted,
        "Cohen Kappa":           kappa,
        "Matthews Corr. Coef.":  mcc,
    }
    for k, v in metrics.items():
        print(f"  {k:<28} {v:.4f}")

    print_section(f"REPORTE POR CLASE{' -- ' + label if label else ''}")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    return metrics


# ──────────────────────────────────────────────
# GRAFICOS
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, classes, output_prefix,
                          suffix="", title_suffix=""):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    n = len(classes)
    cell_size = max(1.2, 5 / n)
    fig, axes = plt.subplots(1, 2, figsize=(cell_size * n * 2 + 2, cell_size * n + 1))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Confusion absoluta{title_suffix}", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) == 0, 0,
                           cm / cm.sum(axis=1, keepdims=True))
    disp_n = ConfusionMatrixDisplay(confusion_matrix=np.round(cm_norm, 2),
                                    display_labels=classes)
    disp_n.plot(ax=axes[1], colorbar=False, cmap="Greens")
    axes[1].set_title(f"Confusion normalizada (por fila){title_suffix}", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = f"{output_prefix}_confusion_matrix{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Guardada: {path}")
    plt.close()


def plot_roc_curves(y_true, y_conf, classes, output_prefix):
    if y_conf is None:
        print("\n  AVISO: Sin columna de confianza -> curvas ROC omitidas.")
        return

    print_section("CURVAS ROC (One-vs-Rest, sin unknown)")
    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fig, ax = plt.subplots(figsize=(8, 6))
    auc_scores = {}
    for i, cls in enumerate(classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_conf)
            auc = roc_auc_score(y_true_bin[:, i], y_conf)
            auc_scores[cls] = auc
            ax.plot(fpr, tpr, label=f"{cls}  (AUC = {auc:.3f})")
        except Exception as e:
            print(f"  AVISO: No se pudo calcular ROC para '{cls}': {e}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC por clase (OvR)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = f"{output_prefix}_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Guardada: {path}")
    plt.close()

    print("\n  AUC por clase:")
    for cls, auc in auc_scores.items():
        print(f"    {cls:<10}  AUC = {auc:.4f}")
    return auc_scores


def plot_precision_recall(y_true, y_conf, classes, output_prefix):
    if y_conf is None:
        return

    print_section("CURVAS PRECISION-RECALL (sin unknown)")
    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        try:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_conf)
            ap = average_precision_score(y_true_bin[:, i], y_conf)
            ax.plot(recall, precision, label=f"{cls}  (AP = {ap:.3f})")
        except Exception as e:
            print(f"  AVISO: No se pudo calcular PR para '{cls}': {e}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall por clase (OvR)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    path = f"{output_prefix}_precision_recall.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Guardada: {path}")
    plt.close()


def plot_class_distribution(y_true, y_pred, classes, output_prefix, unknown_label):
    print_section("DISTRIBUCION DE CLASES")

    has_unk = (y_pred == unknown_label).any()
    all_pred = classes + ([unknown_label] if has_unk else [])
    true_counts = y_true.value_counts().reindex(classes, fill_value=0)
    pred_counts = y_pred.value_counts().reindex(all_pred, fill_value=0)

    for cls in classes:
        print(f"  {cls:<12}  real={true_counts[cls]:>5}  predicho={pred_counts.get(cls, 0):>5}")
    if has_unk:
        print(f"  {unknown_label:<12}  real={'---':>5}  predicho={pred_counts[unknown_label]:>5}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(np.arange(len(classes)), true_counts.values, color="#4C72B0")
    axes[0].set_xticks(np.arange(len(classes)))
    axes[0].set_xticklabels(classes, rotation=20)
    axes[0].set_title("Distribucion real de clases")
    axes[0].set_ylabel("N muestras")
    axes[0].grid(axis="y", alpha=0.3)

    colors = ["#DD8452" if c != unknown_label else "#E74C3C" for c in all_pred]
    axes[1].bar(np.arange(len(all_pred)), pred_counts.values, color=colors)
    axes[1].set_xticks(np.arange(len(all_pred)))
    axes[1].set_xticklabels(all_pred, rotation=20)
    axes[1].set_title("Distribucion predicha (rojo = unknown)")
    axes[1].set_ylabel("N muestras")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = f"{output_prefix}_class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Guardada: {path}")
    plt.close()


def plot_confidence_histogram(y_true, y_pred, y_conf, output_prefix, unknown_label):
    if y_conf is None:
        return

    print_section("DISTRIBUCION DE CONFIANZA")
    mask_unk = y_pred == unknown_label
    correct  = (y_true == y_pred) & ~mask_unk
    wrong    = (y_true != y_pred) & ~mask_unk

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(y_conf[correct.values], bins=30, alpha=0.7, color="green", label="Correctas")
    axes[0].hist(y_conf[wrong.values],   bins=30, alpha=0.7, color="red",   label="Incorrectas")
    if mask_unk.any():
        axes[0].hist(y_conf[mask_unk.values], bins=30, alpha=0.7, color="purple", label="Unknown")
    axes[0].set_title("Confianza por resultado")
    axes[0].set_xlabel("Confianza")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(y_conf, bins=30, color="steelblue", alpha=0.8)
    axes[1].set_title("Distribucion global de confianza")
    axes[1].set_xlabel("Confianza")
    axes[1].set_ylabel("Frecuencia")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{output_prefix}_confidence_hist.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Guardada: {path}")
    plt.close()


def save_metrics_csv(metrics_excl, metrics_incl, classes, y_true, y_pred,
                     output_prefix, unknown_label):
    from sklearn.metrics import precision_score, recall_score, f1_score as f1s

    rows = []
    mask_k = y_pred != unknown_label
    for cls in classes:
        p = precision_score(y_true[mask_k] == cls, y_pred[mask_k] == cls, zero_division=0)
        r = recall_score(   y_true[mask_k] == cls, y_pred[mask_k] == cls, zero_division=0)
        f = f1s(            y_true[mask_k] == cls, y_pred[mask_k] == cls, zero_division=0)
        rows.append({"clase": cls, "precision": round(p, 4),
                     "recall": round(r, 4), "f1": round(f, 4)})

    df_cls    = pd.DataFrame(rows)
    df_global = pd.DataFrame([
        {"metrica": k,
         "excluido_unknown": round(v, 4),
         "incluido_unknown": round(metrics_incl.get(k, float("nan")), 4)}
        for k, v in metrics_excl.items()
    ])

    path_cls    = f"{output_prefix}_metricas_por_clase.csv"
    path_global = f"{output_prefix}_metricas_globales.csv"
    df_cls.to_csv(path_cls,    index=False)
    df_global.to_csv(path_global, index=False)
    print(f"\n  Guardadas: {path_cls}, {path_global}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evalua metricas de clasificacion. Maneja predicciones 'unknown'.")
    parser.add_argument("csv",       nargs="?", default=DEFAULT_CSV)
    parser.add_argument("--label",   default=DEFAULT_LABEL)
    parser.add_argument("--pred",    default=DEFAULT_PRED)
    parser.add_argument("--conf",    default=DEFAULT_CONF)
    parser.add_argument("--unknown", default=DEFAULT_UNKNOWN,
                        help="Etiqueta de prediccion desconocida (default: 'unknown')")
    parser.add_argument("--output",  default=OUTPUT_PREFIX)
    args = parser.parse_args()

    print(f"\n>>>  Cargando: {args.csv}")
    y_true, y_pred, y_conf, df = load_data(args.csv, args.label, args.pred, args.conf)

    known_classes = sorted(y_true.unique().tolist())
    has_unknown   = (y_pred == args.unknown).any()
    print(f"  Muestras : {len(y_true):,}")
    print(f"  Clases   : {known_classes}")
    if has_unknown:
        n_unk = (y_pred == args.unknown).sum()
        print(f"  AVISO: {n_unk} predicciones '{args.unknown}' detectadas")

    # 1. Analisis de unknown
    y_true_k, y_pred_k, y_conf_k, mask_unk = analyze_unknown(
        y_true, y_pred, y_conf, args.unknown)

    if has_unknown:
        plot_unknown_breakdown(y_true, mask_unk, known_classes,
                               args.unknown, args.output)

    # 2. Metricas EXCLUYENDO unknown
    metrics_excl = compute_metrics(y_true_k, y_pred_k, known_classes,
                                   label="unknown EXCLUIDO")

    # 3. Metricas INCLUYENDO unknown (penaliza)
    if has_unknown:
        all_classes_with_unk = known_classes + [args.unknown]
        metrics_incl = compute_metrics(y_true, y_pred, all_classes_with_unk,
                                       label="unknown INCLUIDO (penaliza)")
    else:
        metrics_incl = metrics_excl

    # 4. Distribucion de clases
    plot_class_distribution(y_true, y_pred, known_classes,
                            args.output, args.unknown)

    # 5. Matriz de confusion sin unknown
    print_section("MATRIZ DE CONFUSION -- unknown EXCLUIDO")
    plot_confusion_matrix(y_true_k, y_pred_k, known_classes,
                          args.output, suffix="", title_suffix="")

    # 6. Matriz de confusion con unknown
    if has_unknown:
        print_section("MATRIZ DE CONFUSION -- unknown INCLUIDO")
        plot_confusion_matrix(y_true, y_pred, known_classes + [args.unknown],
                              args.output, suffix="_con_unknown",
                              title_suffix=" (con unknown)")

    # 7. Curvas ROC / PR (siempre sin unknown)
    plot_roc_curves(y_true_k, y_conf_k, known_classes, args.output)
    plot_precision_recall(y_true_k, y_conf_k, known_classes, args.output)

    # 8. Histograma de confianza
    plot_confidence_histogram(y_true, y_pred, y_conf, args.output, args.unknown)

    # 9. Exportar CSVs
    save_metrics_csv(metrics_excl, metrics_incl, known_classes,
                     y_true, y_pred, args.output, args.unknown)

    print(f"\n[OK] Evaluacion completada!\n")


if __name__ == "__main__":
    main()