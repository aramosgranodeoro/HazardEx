"""
evaluar_chartqa.py
--------------------
Evalúa resultados de ChartQA (pregunta-respuesta abierta sobre gráficos),
donde no hay categorías fijas sino una `label` (respuesta correcta) y una
`respuesta_modelo` de texto libre.

Como no es clasificación, no se puede comparar por igualdad exacta de
categorías. Se usa una estrategia en dos pasos:

  1. LIKE: ¿aparece la etiqueta (como palabra completa) dentro de la
     respuesta del modelo, o viceversa? Cubre respuestas cortas y
     respuestas largas que mencionan el valor correcto en algún punto
     ("...el resultado es 14...").

  2. Si la etiqueta es numérica y el LIKE no encontró coincidencia, se
     extraen todos los números de la respuesta del modelo -- en formato
     decimal (con . o ,), porcentaje (25%), fracción (a/b) o escritos en
     palabras (inglés y español, ej. "catorce", "twenty two") -- y se
     compara cada uno con la etiqueta con una tolerancia relativa del 5%
     (igual que la métrica "relaxed accuracy" estándar de ChartQA).
     También se prueban las variantes etiqueta*100 y etiqueta/100 para
     cubrir el caso de que uno esté en escala 0-1 y el otro en 0-100
     (porcentaje vs. fracción).

Limitaciones conocidas:
  - El parser de números en palabras cubre 0-100 en inglés y español
    (unidades, decenas y compuestos tipo "twenty two" / "veinticuatro").
    No cubre números compuestos por encima de 100 escritos en palabras.
  - El LIKE es sensible a acentos/mayúsculas (se normalizan) pero no
    entiende sinónimos ("green line" vs. descripción del color en otro
    idioma), así que puede haber falsos negativos en respuestas
    verborrágicas que no citan la etiqueta literalmente.

Uso:
  python evaluar_chartqa.py --input <carpeta_csvs> --output <carpeta_resultados>
"""

import argparse
import json
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

COST_COLS = ["total_ms", "prompt_ms", "eval_ms", "prompt_tokens", "generated_tokens", "tokens_per_sec"]

COL_CANDIDATES = {
    "label": ["label", "label_real", "etiqueta"],
    "resp":  ["respuesta_modelo", "prediccion", "respuesta"],
}

TOLERANCE = 0.05  # 5% relativo, igual que "relaxed accuracy" de ChartQA

EN_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}
ES_NUMBERS = {
    "cero": 0, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11,
    "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    "veintiuno": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
    "veintinueve": 29, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100,
    "ciento": 100,
}
TENS = {20, 30, 40, 50, 60, 70, 80, 90}


# ──────────────────────────────────────────────
# Extracción de la respuesta real cuando viene embebida en JSON
# ──────────────────────────────────────────────

def extract_primary_answer(text: str) -> str:
    """
    Algunas respuestas de llava_7b_es vienen como un bloque JSON sin parsear
    (```json {"respuesta": ..., "confianza": ..., "razonamiento": ...} ```).
    Si se detecta ese patrón, se usa SOLO el valor de 'respuesta' para el
    matching, para no confundirlo con números de 'confianza' o texto de
    'razonamiento'. Si no hay JSON reconocible, se devuelve el texto tal cual.
    """
    if not isinstance(text, str) or "respuesta" not in text.lower():
        return text

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return text
    blob = m.group()

    try:
        obj = json.loads(blob)
        if isinstance(obj, dict) and "respuesta" in obj:
            return str(obj["respuesta"])
    except Exception:
        pass

    m2 = re.search(r'"respuesta"\s*:\s*"([^"]*)"', blob)
    if m2:
        return m2.group(1)
    m3 = re.search(r'"respuesta"\s*:\s*(-?[\d.,]+)', blob)
    if m3:
        return m3.group(1)

    return text


# ──────────────────────────────────────────────
# Normalización de texto
# ──────────────────────────────────────────────

def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_text(s) -> str:
    if pd.isna(s):
        return ""
    s = strip_accents(str(s)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ──────────────────────────────────────────────
# Parsing numérico
# ──────────────────────────────────────────────

def try_parse_number(s) -> float | None:
    """Intenta parsear la etiqueta (normalmente un valor 'limpio') como número."""
    if pd.isna(s):
        return None
    txt = str(s).strip().replace("%", "")
    # fracción a/b
    m = re.fullmatch(r"(-?\d+(?:[.,]\d+)?)\s*/\s*(-?\d+(?:[.,]\d+)?)", txt)
    if m:
        try:
            a = float(m.group(1).replace(",", "."))
            b = float(m.group(2).replace(",", "."))
            return a / b if b != 0 else None
        except Exception:
            return None
    txt = txt.replace(",", ".")
    try:
        return float(txt)
    except Exception:
        return None


def extract_word_numbers(text_norm: str) -> set[float]:
    words = re.findall(r"[a-z]+", text_norm)
    candidates: set[float] = set()
    for w in words:
        if w in EN_NUMBERS:
            candidates.add(float(EN_NUMBERS[w]))
        if w in ES_NUMBERS:
            candidates.add(float(ES_NUMBERS[w]))
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        for table in (EN_NUMBERS, ES_NUMBERS):
            if a in table and table[a] in TENS and b in table and table[b] < 10:
                candidates.add(float(table[a] + table[b]))
    return candidates


def extract_numeric_candidates(text_norm: str) -> set[float]:
    candidates: set[float] = set()

    # fracciones a/b
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)", text_norm):
        try:
            a = float(m.group(1).replace(",", "."))
            b = float(m.group(2).replace(",", "."))
            if b != 0:
                candidates.add(a / b)
        except Exception:
            pass

    # números sueltos, con o sin % (o "percent"/"por ciento" escrito en palabras)
    for m in re.finditer(r"-?\d+(?:[.,]\d+)?\s*%?", text_norm):
        tok = m.group()
        is_pct = "%" in tok
        # el modelo a veces escribe "56 percent" / "56 por ciento" en vez de "56%"
        if not is_pct:
            tail = text_norm[m.end():m.end() + 15]
            if re.match(r"\s*(percent|por\s*ciento|porciento)", tail):
                is_pct = True
        tok_clean = tok.replace("%", "").strip()
        if "," in tok_clean and "." not in tok_clean:
            parts = tok_clean.split(",")
            if len(parts) == 2 and len(parts[1]) <= 3:
                tok_clean = parts[0] + "." + parts[1]
            else:
                tok_clean = tok_clean.replace(",", "")
        else:
            tok_clean = tok_clean.replace(",", "")
        try:
            val = float(tok_clean)
        except Exception:
            continue
        candidates.add(val)
        if is_pct:
            candidates.add(val / 100)

    candidates |= extract_word_numbers(text_norm)
    return candidates


def numeric_isclose(gold: float, pred: float, tol: float = TOLERANCE) -> bool:
    if gold == 0:
        return abs(pred) < 1e-6
    return abs(pred - gold) <= tol * abs(gold)


def numeric_match(label_num: float, candidates: set[float]) -> bool:
    for c in candidates:
        if numeric_isclose(label_num, c):
            return True
    # escala 0-1 vs 0-100 (fracción vs porcentaje)
    if 0 <= label_num <= 1:
        for c in candidates:
            if numeric_isclose(label_num * 100, c):
                return True
    if label_num > 1:
        for c in candidates:
            if numeric_isclose(label_num / 100, c):
                return True
    return False


# ──────────────────────────────────────────────
# Matching principal
# ──────────────────────────────────────────────

def is_correct(label, response) -> tuple[bool, str]:
    label_norm = normalize_text(label)

    response_effective = extract_primary_answer(response) if isinstance(response, str) else response
    resp_norm  = normalize_text(response_effective)

    if not label_norm or not resp_norm:
        return False, "vacio"

    # 1. LIKE (palabra completa) en ambas direcciones
    pattern_label = r"(?<!\w)" + re.escape(label_norm) + r"(?!\w)"
    if re.search(pattern_label, resp_norm):
        return True, "like"
    if len(resp_norm) <= len(label_norm) + 5:
        pattern_resp = r"(?<!\w)" + re.escape(resp_norm) + r"(?!\w)"
        if re.search(pattern_resp, label_norm):
            return True, "like_reverse"

    # 2. Numérico (fracción / palabra / decimal / porcentaje)
    label_num = try_parse_number(label)
    if label_num is not None:
        candidates = extract_numeric_candidates(resp_norm)
        if numeric_match(label_num, candidates):
            return True, "numerico"

    return False, "sin_match"


# ──────────────────────────────────────────────
# Helpers de carga / columnas
# ──────────────────────────────────────────────

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


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_cost_metrics(df):
    cost = {}
    for c in COST_COLS:
        if c in df.columns:
            cost[f"mean_{c}"] = round(pd.to_numeric(df[c], errors="coerce").mean(), 2)
        else:
            cost[f"mean_{c}"] = np.nan
    return cost


def determine_language(model_name: str) -> str:
    name_lower = model_name.lower()
    if "salamandra" in name_lower:
        return "es"
    if re.search(r"_es(\b|_|$)", name_lower):
        return "es"
    return "en"


def base_model_name(model_name: str) -> str:
    return re.sub(r"_es$", "", model_name, flags=re.IGNORECASE)


def plot_accuracy(df_summary, output_folder):
    df_plot = df_summary.sort_values("model", key=lambda s: s.str.lower())
    models = df_plot["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 2.2), 5))
    b1 = ax.bar(x - width / 2, df_plot["accuracy"].tolist(), width,
                color="#4C72B0", label="Accuracy")
    b2 = ax.bar(x + width / 2, df_plot["accuracy_penalizada"].tolist(), width,
                color="#DD8452", label="Accuracy penalizada (vacías = fallo)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_title("ChartQA — Accuracy por modelo (LIKE + comparación numérica)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = output_folder / "comparacion_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico de accuracy guardado en: {path}")


def plot_match_methods(df_all_rows, output_folder):
    """Barras apiladas: qué % de aciertos vino de LIKE vs numérico, por modelo."""
    piv = (
        df_all_rows[df_all_rows["es_correcto"]]
        .groupby(["model", "metodo_match"]).size().unstack(fill_value=0)
    )
    if piv.empty:
        return
    piv = piv.div(df_all_rows.groupby("model").size(), axis=0)
    piv = piv.sort_index(key=lambda s: s.str.lower())

    fig, ax = plt.subplots(figsize=(max(7, len(piv) * 2.2), 5))
    bottom = np.zeros(len(piv))
    colors = {"like": "#4C72B0", "like_reverse": "#8CB4E8", "numerico": "#DD8452"}
    for method in piv.columns:
        vals = piv[method].values
        ax.bar(piv.index, vals, bottom=bottom, label=method,
              color=colors.get(method, "#999999"))
        bottom += vals

    ax.set_ylabel("Proporción de muestras")
    ax.set_title("ChartQA — Origen de los aciertos (LIKE vs. numérico)",
                 fontsize=13, fontweight="bold")
    ax.set_xticklabels(piv.index, rotation=20, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = output_folder / "comparacion_metodo_match.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico de método de match guardado en: {path}")


def build_language_comparison(all_metrics: list[dict], output_folder: Path) -> None:
    if not all_metrics:
        return
    df = pd.DataFrame(all_metrics)
    df["lang"] = df["model"].apply(determine_language)
    df["base_model"] = df["model"].apply(base_model_name)

    metric_cols = ["accuracy", "accuracy_penalizada"] + [f"mean_{c}" for c in COST_COLS]
    available = [c for c in metric_cols if c in df.columns]

    rows = []
    for base in sorted(df["base_model"].unique()):
        sub = df[df["base_model"] == base]
        row = {"base_model": base}
        for lang in ["es", "en"]:
            lang_sub = sub[sub["lang"] == lang]
            row[f"model_{lang}"] = " | ".join(lang_sub["model"].tolist()) or "—"
            for m in available:
                row[f"{m}_{lang}"] = round(lang_sub[m].mean(), 4) if not lang_sub.empty and not lang_sub[m].isna().all() else np.nan
        for m in ["accuracy", "accuracy_penalizada"]:
            es_v, en_v = row.get(f"{m}_es"), row.get(f"{m}_en")
            row[f"{m}_diff_es_minus_en"] = round(es_v - en_v, 4) if pd.notna(es_v) and pd.notna(en_v) else np.nan
        rows.append(row)

    df_cmp = pd.DataFrame(rows)
    out_path = output_folder / "comparacion_es_vs_en.csv"
    df_cmp.to_csv(out_path, index=False)
    print(f"\n  Comparativa ES vs EN guardada en: {out_path}")
    print(df_cmp.to_string(index=False))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evalúa modelos en ChartQA (LIKE + comparación numérica).")
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", default="resultados_evaluacion_chartqa")
    return parser.parse_args()


def main():
    args = parse_args()
    input_folder  = Path(args.input).resolve()
    output_folder = Path(args.output).resolve()

    if not input_folder.exists():
        print(f"[ERROR] La carpeta de entrada no existe: {input_folder}", file=sys.stderr)
        sys.exit(1)

    csv_files = find_csvs(input_folder)
    if not csv_files:
        print(f"[ERROR] No se encontraron CSVs en: {input_folder}", file=sys.stderr)
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_rows_frames = []

    for csv_path in csv_files:
        model_name = csv_path.stem
        lang = determine_language(model_name)
        print(f"\n{'─'*60}")
        print(f"  Modelo: {model_name}  [idioma detectado: {lang.upper()}]")
        print(f"{'─'*60}")

        try:
            df_raw = load_csv(csv_path)
        except Exception as e:
            print(f"  [SKIP] No se pudo cargar: {e}")
            continue

        col_label = resolve_column(df_raw, COL_CANDIDATES["label"])
        col_resp  = resolve_column(df_raw, COL_CANDIDATES["resp"])
        if col_label is None or col_resp is None:
            print(f"  [SKIP] No se encontraron columnas de label/respuesta. Columnas: {list(df_raw.columns)}")
            continue

        print(f"  Filas cargadas: {len(df_raw)}  |  Columnas -> label: '{col_label}' | respuesta: '{col_resp}'")

        n_empty = int(df_raw[col_resp].isna().sum() + (df_raw[col_resp].astype(str).str.strip() == "").sum())

        results = df_raw.apply(
            lambda r: is_correct(r[col_label], r[col_resp]), axis=1, result_type="expand"
        )
        results.columns = ["es_correcto", "metodo_match"]
        df_out = df_raw.copy()
        df_out["es_correcto"]   = results["es_correcto"]
        df_out["metodo_match"]  = results["metodo_match"]
        df_out["model"] = model_name

        n_total   = len(df_out)
        n_correct = int(df_out["es_correcto"].sum())
        accuracy  = round(n_correct / n_total, 4) if n_total else np.nan
        # accuracy_penalizada: aquí no hay predicciones "inválidas" separadas de las
        # incorrectas (una respuesta vacía ya cuenta como fallo), así que coincide
        # con accuracy salvo que existan respuestas vacías, que se reportan aparte.
        accuracy_penalizada = accuracy

        pct_empty = round(n_empty / n_total * 100, 2) if n_total else 0

        print(f"  Respuestas vacías: {n_empty} ({pct_empty}%)")
        print(f"  Accuracy           : {accuracy}")
        print(f"  Accuracy penalizada: {accuracy_penalizada}")
        print(f"  Distribución método de match (sobre aciertos):")
        print(df_out.loc[df_out["es_correcto"], "metodo_match"].value_counts().to_string())

        cost_metrics = compute_cost_metrics(df_out)

        metrics = {
            "model": model_name,
            "language": lang,
            "csv_file": csv_path.name,
            "n_samples": n_total,
            "n_correct": n_correct,
            "n_empty": n_empty,
            "pct_empty": pct_empty,
            "accuracy": accuracy,
            "accuracy_penalizada": accuracy_penalizada,
            **cost_metrics,
        }
        all_metrics.append(metrics)

        model_dir = output_folder / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        detail_path = model_dir / "detalle_predicciones.csv"
        cols_out = ["indice"] if "indice" in df_out.columns else []
        cols_out += [c for c in [col_label, col_resp, "es_correcto", "metodo_match"] if c in df_out.columns]
        df_out[cols_out].to_csv(detail_path, index=False)
        print(f"  Detalle guardado en: {detail_path.relative_to(output_folder.parent)}")

        all_rows_frames.append(df_out[["model", "es_correcto", "metodo_match"]])

    if not all_metrics:
        print("\n[WARN] No se generaron métricas para ningún modelo.")
        return

    df_summary = pd.DataFrame(all_metrics).sort_values("accuracy_penalizada", ascending=False)
    summary_path = output_folder / "comparacion_modelos.csv"
    df_summary.to_csv(summary_path, index=False)

    print(f"\n{'═'*60}")
    print(f"  Resumen guardado en: {summary_path}")
    print(f"{'═'*60}")
    print(df_summary.to_string(index=False))

    plot_accuracy(df_summary, output_folder)

    if all_rows_frames:
        plot_match_methods(pd.concat(all_rows_frames, ignore_index=True), output_folder)

    build_language_comparison(all_metrics, output_folder)

    print("\n[OK] Evaluación completada.\n")


if __name__ == "__main__":
    main()