"""
evaluar_modelos.py
------------------
Recorre una carpeta en busca de archivos CSV con resultados de modelos de
clasificación (columnas esperadas: label_real, prediccion, correcto, confianza,
total_ms, prompt_ms, eval_ms, prompt_tokens, generated_tokens, tokens_per_sec).

Categorías soportadas (6):
  violence / violencia · weapons / armas · fire / fuego
  accident / accidente · chart / gráfico / grafico · normal

Por cada CSV:
  - Normaliza etiquetas EN↔ES al espacio canónico en español.
  - Calcula métricas de clasificación: accuracy, precision, recall, f1,
    AUC-ROC (macro OvR), MCC y porcentaje de predicciones inválidas.
  - Calcula métricas de coste computacional (media por muestra).
  - Genera una carpeta por modelo con la matriz de confusión normalizada
    (clases ordenadas alfabéticamente).

Al finalizar:
  - Guarda un CSV resumen comparando todos los modelos (métricas de clasificación).
  - Gráfico comparativo de métricas de clasificación (modelos en orden alfabético).
  - Gráfico comparativo de métricas de coste computacional (ídem).
  - CSV comparativa ES vs no-ES por modelo (según nombre de archivo):
      · Sufijo _es  → versión en español
      · "salamandra" en el nombre → siempre español
      · Sin _es y sin "salamandra" → versión en inglés / base

Uso:
  python evaluar_modelos.py --input <carpeta_csvs> --output <carpeta_resultados>

Argumentos opcionales:
  --col_real       Nombre columna con etiqueta real     (default: label_real)
  --col_pred       Nombre columna con predicción        (default: prediccion)
  --col_conf       Nombre columna con confianza/prob    (default: confianza)
  --invalid_label  Valor que indica predicción inválida (default: ERROR)
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# Columnas de coste computacional esperadas
COST_COLS = ["total_ms", "prompt_ms", "eval_ms", "prompt_tokens", "generated_tokens", "tokens_per_sec"]

# Etiquetas canónicas (en español) en orden alfabético
CANONICAL_CLASSES = sorted(["accidente", "armas", "fuego", "grafico", "normal", "violencia"])

# Mapa de normalización de etiquetas EN/ES → etiqueta canónica
# Incluye variantes con y sin tilde para robustez.
LABEL_MAP = {
    # Violencia / Violence
    "violence":      "violencia",
    "violencia":     "violencia",
    "fight":         "violencia",
    # Armas / Weapons
    "weapons":       "armas",
    "weapon":        "armas",
    "armas":         "armas",
    "arma":          "armas",
    # Fuego / Fire
    "fire":          "fuego",
    "fuego":         "fuego",
    "smoke":         "fuego",   # fuego/humo se agrupa en la misma categoría
    # Accidente / Accident
    "accident":      "accidente",
    "accidente":     "accidente",
    # Gráfico / Chart — con y sin tilde
    "chart":         "grafico",
    "charts":        "grafico",
    "grafico":       "grafico",
    "gráfico":       "grafico",
    "disinformation": "grafico",
    # Normal
    "normal":        "normal",
    "no_violence":   "normal",
    "non-fight":     "normal",
    "non_fight":     "normal",
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa modelos de clasificación a partir de CSVs de resultados."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Carpeta con los CSVs de resultados.")
    parser.add_argument("--output", "-o", default="resultados_evaluacion",
                        help="Carpeta raíz de salida (default: resultados_evaluacion).")
    parser.add_argument("--col_real",      default="label_real")
    parser.add_argument("--col_pred",      default="prediccion")
    parser.add_argument("--col_conf",      default="confianza")
    parser.add_argument("--invalid_label", default="ERROR")
    parser.add_argument(
        "--order", nargs="+", default=None,
        help="Orden de procesamiento: lista de stems de CSV sin extensión."
    )
    return parser.parse_args()


def find_csvs(folder: Path) -> list[Path]:
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        csvs = sorted(folder.rglob("*.csv"))
    return csvs


def load_csv(path: Path) -> pd.DataFrame:
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise ValueError(f"No se pudo leer '{path}' con separadores estándar.")


def normalize_label(val: str) -> str:
    """Normaliza una etiqueta al espacio canónico en español."""
    v = str(val).strip().lower()
    return LABEL_MAP.get(v, v)


def clean_df(df, col_real, col_pred, col_conf, invalid_label):
    for col in [col_real, col_pred]:
        if col not in df.columns:
            raise KeyError(
                f"Columna obligatoria '{col}' no encontrada. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    n_invalidas = int(
        (df[col_pred].astype(str).str.upper() == invalid_label.upper()).sum()
    )
    df_clean = df[df[col_pred].astype(str).str.upper() != invalid_label.upper()].copy()

    if col_conf not in df_clean.columns:
        df_clean[col_conf] = np.nan

    df_clean[col_real] = df_clean[col_real].astype(str).str.strip().apply(normalize_label)
    df_clean[col_pred] = df_clean[col_pred].astype(str).str.strip().apply(normalize_label)
    df_clean[col_conf] = pd.to_numeric(df_clean[col_conf], errors="coerce")

    for c in COST_COLS:
        if c in df_clean.columns:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

    return df_clean, n_invalidas


def compute_metrics(y_true, y_pred, n_total=None):
    """
    Dataset balanceado: la métrica principal es accuracy.
    n_total: filas originales incluyendo ERRORs; si se pasa se calcula
    accuracy_penalizada = aciertos / n_total (los ERRORs cuentan como fallos).
    """
    classes = sorted(y_true.unique().tolist())

    m = {}
    m["n_samples"]  = len(y_true)
    m["n_classes"]  = len(classes)
    m["classes"]    = "|".join(classes)
    m["accuracy"]   = round(accuracy_score(y_true, y_pred), 4)

    if n_total and n_total > 0:
        n_correct = int((y_true == y_pred).sum())
        m["accuracy_penalizada"] = round(n_correct / n_total, 4)
    else:
        m["accuracy_penalizada"] = m["accuracy"]

    return m


def compute_cost_metrics(df):
    """Calcula la media de cada métrica de coste computacional disponible."""
    cost = {}
    for c in COST_COLS:
        if c in df.columns:
            cost[f"mean_{c}"] = round(df[c].mean(), 2)
        else:
            cost[f"mean_{c}"] = np.nan
    return cost


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Genera la matriz de confusión normalizada con clases en orden alfabético."""
    # Unión de clases reales + predichas, siempre en orden alfabético
    present = sorted(set(y_true.unique().tolist()) | set(y_pred.unique().tolist()))
    classes = present  # ya ordenadas

    cm      = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(7, 6))
    im      = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar    = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proporción", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=9,
                    color="white" if val > 0.5 else "black")

    ax.set_ylabel("Etiqueta real", fontsize=12)
    ax.set_xlabel("Predicción",    fontsize=12)
    ax.set_title(f"Matriz de Confusión — {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = output_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_classification_metrics(df_summary, output_folder):
    """Gráfico de barras: accuracy vs accuracy_penalizada por modelo (orden alfabético)."""
    metric_cols = ["accuracy", "accuracy_penalizada"]
    available   = [c for c in metric_cols
                   if c in df_summary.columns and df_summary[c].notna().any()]
    if len(df_summary) < 1 or not available:
        return

    df_plot = df_summary.sort_values("model", key=lambda s: s.str.lower())
    models  = df_plot["model"].tolist()
    x       = np.arange(len(models))
    width   = 0.35

    colors = ["#4C72B0", "#DD8452"]
    labels_map = {
        "accuracy":            "Accuracy",
        "accuracy_penalizada": "Accuracy penalizada (ERRORs = fallo)",
    }

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 2.2), 5))

    for k, col in enumerate(available):
        vals = df_plot[col].fillna(0).tolist()
        offset = (k - (len(available) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width,
                      color=colors[k], label=labels_map.get(col, col))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy por modelo (dataset balanceado)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = output_folder / "comparacion_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico de accuracy guardado en: {path}")


def plot_cost_metrics(df_summary, output_folder):
    """Una imagen independiente por cada métrica de coste. Modelos en orden alfabético."""
    cost_cols_mean = [f"mean_{c}" for c in COST_COLS]
    available = [c for c in cost_cols_mean
                 if c in df_summary.columns and df_summary[c].notna().any()]
    if not available:
        print("  [WARN] No hay métricas de coste disponibles para graficar.")
        return

    df_plot = df_summary.sort_values("model", key=lambda s: s.str.lower())
    models  = df_plot["model"].tolist()
    colors  = plt.cm.tab10(np.linspace(0, 0.6, len(models)))

    # Título en español, unidad y nombre de archivo por métrica
    meta = {
        "mean_total_ms":         ("Tiempo total por muestra (ms)",           "ms",      "coste_tiempo_total"),
        "mean_prompt_ms":        ("Tiempo de prompt por muestra (ms)",        "ms",      "coste_tiempo_prompt"),
        "mean_eval_ms":          ("Tiempo de evaluación por muestra (ms)",    "ms",      "coste_tiempo_eval"),
        "mean_prompt_tokens":    ("Tokens de prompt por muestra",             "tokens",  "coste_tokens_prompt"),
        "mean_generated_tokens": ("Tokens generados por muestra",             "tokens",  "coste_tokens_generados"),
        "mean_tokens_per_sec":   ("Velocidad de generación (tokens/segundo)", "tok/s",   "coste_velocidad"),
    }

    for col in available:
        titulo, unidad, fname = meta.get(col, (col, "", col))
        vals = df_plot[col].fillna(0).tolist()

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.8), 4.5))
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor="white")

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:,.1f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=18, ha="right", fontsize=9)
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_ylabel(unidad, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
        plt.tight_layout()

        path = output_folder / f"{fname}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {path}")


def determine_language(model_name: str) -> str:
    """
    Devuelve 'es' si el nombre de archivo termina en _es
    o contiene 'salamandra'. En cualquier otro caso devuelve 'en'.
    """
    name_lower = model_name.lower()
    if "salamandra" in name_lower:
        return "es"
    if re.search(r"_es(\b|_|$)", name_lower):
        return "es"
    return "en"


def base_model_name(model_name: str) -> str:
    """
    Extrae el nombre base del modelo eliminando el sufijo _es y variantes.
    Ej: resultados_llava_7_es → resultados_llava_7
    """
    base = re.sub(r"_es$", "", model_name, flags=re.IGNORECASE)
    return base


def build_language_comparison(all_metrics: list[dict], output_folder: Path) -> None:
    """
    Genera un CSV que compara métricas de clasificación y coste
    entre la versión en español (ES) y en inglés/base (EN) de cada modelo.
    """
    if not all_metrics:
        return

    df = pd.DataFrame(all_metrics)
    df["lang"]       = df["model"].apply(determine_language)
    df["base_model"] = df["model"].apply(base_model_name)

    metric_cols = (
        ["accuracy", "accuracy_penalizada"]
        + [f"mean_{c}" for c in COST_COLS]
    )
    available_metrics = [c for c in metric_cols if c in df.columns]

    rows = []
    for base in sorted(df["base_model"].unique()):
        sub = df[df["base_model"] == base]
        row = {"base_model": base}

        for lang in ["es", "en"]:
            lang_sub  = sub[sub["lang"] == lang]
            model_names = lang_sub["model"].tolist()
            row[f"model_{lang}"] = " | ".join(model_names) if model_names else "—"
            for m in available_metrics:
                if m in lang_sub.columns and not lang_sub[m].isna().all():
                    row[f"{m}_{lang}"] = round(lang_sub[m].mean(), 4)
                else:
                    row[f"{m}_{lang}"] = np.nan

        clf_metrics = ["accuracy", "accuracy_penalizada"]
        for m in clf_metrics:
            es_val = row.get(f"{m}_es", np.nan)
            en_val = row.get(f"{m}_en", np.nan)
            if pd.notna(es_val) and pd.notna(en_val):
                row[f"{m}_diff_es_minus_en"] = round(es_val - en_val, 4)
            else:
                row[f"{m}_diff_es_minus_en"] = np.nan

        rows.append(row)

    df_cmp = pd.DataFrame(rows)

    col_order = ["base_model", "model_es", "model_en"]
    for m in available_metrics:
        col_order += [f"{m}_es", f"{m}_en"]
    clf_metrics_diff = ["accuracy", "accuracy_penalizada"]
    for m in clf_metrics_diff:
        col_order.append(f"{m}_diff_es_minus_en")

    col_order = [c for c in col_order if c in df_cmp.columns]
    df_cmp    = df_cmp[col_order]

    out_path = output_folder / "comparacion_es_vs_en.csv"
    df_cmp.to_csv(out_path, index=False)
    print(f"\n  Comparativa ES vs EN guardada en: {out_path}")
    print(df_cmp.to_string(index=False))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    input_folder  = Path(args.input).resolve()
    output_folder = Path(args.output).resolve()

    if not input_folder.exists():
        print(f"[ERROR] La carpeta de entrada no existe: {input_folder}", file=sys.stderr)
        sys.exit(1)

    csv_files = find_csvs(input_folder)
    if not csv_files:
        print(f"[ERROR] No se encontraron archivos CSV en: {input_folder}", file=sys.stderr)
        sys.exit(1)

    if args.order:
        order_map = {stem: i for i, stem in enumerate(args.order)}
        def sort_key(p):
            for stem, idx in order_map.items():
                if stem in p.stem:
                    return idx
            return len(args.order)
        csv_files = sorted(csv_files, key=sort_key)

    print(f"[INFO] Encontrados {len(csv_files)} CSV(s) en '{input_folder}'")
    output_folder.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []

    for csv_path in csv_files:
        model_name = csv_path.stem
        lang       = determine_language(model_name)
        print(f"\n{'─'*60}")
        print(f"  Modelo: {model_name}  [idioma detectado: {lang.upper()}]")
        print(f"{'─'*60}")

        try:
            df_raw = load_csv(csv_path)
        except Exception as e:
            print(f"  [SKIP] No se pudo cargar: {e}")
            continue

        print(f"  Filas cargadas: {len(df_raw)}")

        try:
            df, n_inv = clean_df(
                df_raw,
                col_real=args.col_real,
                col_pred=args.col_pred,
                col_conf=args.col_conf,
                invalid_label=args.invalid_label,
            )
        except KeyError as e:
            print(f"  [SKIP] {e}")
            continue

        pct_inv = round(n_inv / len(df_raw) * 100, 2) if len(df_raw) > 0 else 0
        print(f"  Predicciones inválidas ('{args.invalid_label}'): {n_inv} ({pct_inv}%)")
        print(f"  Muestras válidas para evaluación: {len(df)}")

        # Diagnóstico: etiquetas no reconocidas (no mapeadas) en predicciones
        known = set(LABEL_MAP.values())
        unknown_preds = df[args.col_pred][~df[args.col_pred].isin(known)].unique().tolist()
        if unknown_preds:
            print(f"  [WARN] Predicciones fuera del mapa de etiquetas: {unknown_preds}")

        if len(df) == 0:
            print("  [SKIP] Sin muestras válidas tras filtrar errores.")
            continue

        y_true = df[args.col_real]
        y_pred = df[args.col_pred]
        y_prob = df[args.col_conf]

        # ── Métricas de clasificación ─────────────────────
        metrics = compute_metrics(y_true, y_pred, n_total=len(df_raw))

        # ── Métricas de coste computacional ──────────────
        cost_metrics = compute_cost_metrics(df)
        metrics.update(cost_metrics)

        metrics["model"]       = model_name
        metrics["language"]    = lang
        metrics["csv_file"]    = csv_path.name
        metrics["n_invalid"]   = n_inv
        metrics["pct_invalid"] = pct_inv

        print(f"  Accuracy           : {metrics['accuracy']}")
        print(f"  Accuracy penalizada: {metrics['accuracy_penalizada']}  (ERRORs como fallos)")
        print(f"  Total ms           : {metrics.get('mean_total_ms', 'N/A')}")
        print(f"  Tok/sec            : {metrics.get('mean_tokens_per_sec', 'N/A')}")

        # ── Carpeta por modelo ────────────────────────────
        model_dir = output_folder / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        report_str  = classification_report(y_true, y_pred, zero_division=0)
        report_path = model_dir / "classification_report.txt"
        report_path.write_text(
            f"Modelo  : {model_name}\n"
            f"Idioma  : {lang.upper()}\n"
            f"Archivo : {csv_path.name}\n"
            f"Muestras válidas: {len(df)} / {len(df_raw)}\n"
            f"Predicciones inválidas ('{args.invalid_label}'): {n_inv} ({pct_inv}%)\n\n"
            + report_str,
            encoding="utf-8",
        )

        cm_path = plot_confusion_matrix(y_true, y_pred, model_name, model_dir)
        print(f"  Matriz de confusión guardada en: {cm_path.relative_to(output_folder.parent)}")

        all_metrics.append(metrics)

    # ── CSV resumen global ────────────────────────────────
    if not all_metrics:
        print("\n[WARN] No se generaron métricas para ningún modelo.")
        return

    col_order = [
        "model", "language", "csv_file",
        "n_samples", "n_invalid", "pct_invalid",
        "n_classes", "classes",
        "accuracy", "accuracy_penalizada",
        "mean_total_ms", "mean_prompt_ms", "mean_eval_ms",
        "mean_prompt_tokens", "mean_generated_tokens", "mean_tokens_per_sec",
    ]
    df_summary = pd.DataFrame(all_metrics)
    col_order_present = [c for c in col_order if c in df_summary.columns]
    df_summary = df_summary[col_order_present].sort_values("accuracy_penalizada", ascending=False)

    summary_path = output_folder / "comparacion_modelos.csv"
    df_summary.to_csv(summary_path, index=False)

    print(f"\n{'═'*60}")
    print(f"  Resumen guardado en: {summary_path}")
    print(f"{'═'*60}")
    print(df_summary.to_string(index=False))

    # ── Gráficos ──────────────────────────────────────────
    plot_classification_metrics(df_summary, output_folder)
    plot_cost_metrics(df_summary, output_folder)

    # ── CSV comparativa ES vs EN ──────────────────────────
    build_language_comparison(all_metrics, output_folder)

    print("\n[OK] Evaluación completada.\n")


if __name__ == "__main__":
    main()