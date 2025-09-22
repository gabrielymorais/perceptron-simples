# atividade05_rede_ps.py
# ------------------------------------------------------------
# Atividade 05 – Rede PS (Perceptron) com o dataset Dermatology
# Requisitos atendidos:
# - 50 rodadas de treino/teste (split estratificado 70/30) -- configurável por CLI
# - Acurácia média e variância
# - Precisão (macro e weighted) e acurácia
# - Matriz de confusão (soma e normalizada)
# - Seleção de atributos: apenas clínicos; apenas histopatológicos
# - Importância dos atributos (magnitude média dos pesos)
# - Saída multiclasse via One-vs-Rest (rótulos em {1..6}); observação {0,1} contemplada
# ------------------------------------------------------------

import os
import sys
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score
)

# -------------------------------
# CLI
# -------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Atividade 05 – Rede PS (Perceptron) no dataset Dermatology"
    )
    p.add_argument("--data", type=str, default="dermatology.data",
                   help="Arquivo de dados (default: dermatology.data)")
    p.add_argument("--names", type=str, default="dermatology.names",
                   help="Arquivo .names opcional (default: dermatology.names)")
    p.add_argument("--out-dir", type=str, default="saida_atividade05",
                   help="Diretório de saída (default: saida_atividade05)")
    p.add_argument("--n-runs", type=int, default=50,
                   help="Número de rodadas (default: 50)")
    p.add_argument("--test-size", type=float, default=0.30,
                   help="Proporção de teste (default: 0.30)")
    p.add_argument("--seed", type=int, default=2025,
                   help="Semente base (default: 2025)")
    p.add_argument("--plot-heatmap", action="store_true",
                   help="Se presente, salva heatmap da matriz de confusão normalizada")
    return p

# -------------------------------
# Utilidades
# -------------------------------
def load_dermatology_csv(path: Path) -> pd.DataFrame:
    """
    Carrega o dataset Dermatology no formato CSV (UCI), tratando '?' como NaN.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    # UCI costuma ser separado por vírgula; header ausente
    df = pd.read_csv(path, header=None, na_values=["?"])
    return df

def infer_class_column(df: pd.DataFrame) -> int:
    """
    Infere qual coluna é a classe (última ou penúltima), retornando o índice inteiro.
    """
    last_unique = df.iloc[:, -1].nunique(dropna=True)
    if 2 <= last_unique <= 10:
        return df.columns[-1]
    sec_last_unique = df.iloc[:, -2].nunique(dropna=True)
    if 2 <= sec_last_unique <= 10:
        return df.columns[-2]
    # Fallback: assumir última
    return df.columns[-1]

def try_parse_feature_names(names_path: Path, n_features: int):
    """
    Tenta extrair nomes de atributos de 'dermatology.names'.
    Se não bater a contagem, retorna nomes genéricos.
    """
    if not names_path.exists():
        return [f"attr_{i+1}" for i in range(n_features)]

    try:
        txt = names_path.read_text(errors="ignore")
    except Exception:
        return [f"attr_{i+1}" for i in range(n_features)]

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    attrs = []
    for ln in lines:
        m = re.match(r'^\s*(\d+)\s*[\.\)]\s*(.+)$', ln)
        if m:
            idx = int(m.group(1))
            name = m.group(2)
            name = re.split(r' - |: ', name)[0].strip()
            attrs.append((idx, name))

    attrs_sorted = sorted(attrs, key=lambda x: x[0])
    names = [nm for i, nm in attrs_sorted if i <= n_features]
    if len(names) == n_features:
        return names
    return [f"attr_{i+1}" for i in range(n_features)]

def split_feature_groups(feature_names):
    """
    Define grupos de atributos:
    - Clínicos: 12 primeiros
    - Histopatológicos: 13º ao 34º (0-based: 12..33)
    Se o total for menor que 34, divide proporcionalmente.
    """
    n = len(feature_names)
    if n >= 34:
        clin_idx = list(range(0, 12))
        hist_idx = list(range(12, 34))
    else:
        k = max(1, int(round(n * (12/34))))
        clin_idx = list(range(0, k))
        hist_idx = list(range(k, n))

    clin_cols = [feature_names[i] for i in clin_idx if i < n]
    hist_cols = [feature_names[i] for i in hist_idx if i < n]
    return clin_cols, hist_cols

def run_experiment(X: pd.DataFrame, y: pd.Series,
                   n_runs=50, test_size=0.3, seed_base=42):
    """
    Executa N rodadas de treino/teste com Perceptron (OvR) + StandardScaler.
    Retorna métricas, confusão acumulada e importância média por atributo.
    """
    accs = []
    p_macros = []
    p_weighteds = []
    conf_sum = None
    feat_importance_accum = None

    classes_sorted = np.sort(y.unique())

    for i in range(n_runs):
        rs = seed_base + i
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=rs
        )
        # StandardScaler (dados densos): with_mean=True (default)
        pipe = make_pipeline(
            StandardScaler(),
            Perceptron(max_iter=1000, tol=1e-3, random_state=rs)
        )
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        p_macros.append(precision_score(y_te, y_pred, average="macro", zero_division=0))
        p_weighteds.append(precision_score(y_te, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_te, y_pred, labels=classes_sorted)
        conf_sum = cm if conf_sum is None else conf_sum + cm

        # Importância = norma L2 dos coeficientes ao longo das classes (OvR)
        percep = pipe.named_steps["perceptron"]
        coefs = percep.coef_  # shape: (n_classes, n_features)
        importance = np.linalg.norm(coefs, axis=0)  # shape: (n_features,)
        feat_importance_accum = importance if feat_importance_accum is None else feat_importance_accum + importance

    feat_importance_mean = feat_importance_accum / n_runs

    results = {
        "accs": np.array(accs),
        "prec_macros": np.array(p_macros),
        "prec_weighteds": np.array(p_weighteds),
        "confusion_sum": conf_sum,
        "classes": classes_sorted,
        "feat_importance": feat_importance_mean
    }
    return results

def summarize(label, res) -> dict:
    return {
        "Configuração": label,
        "Acurácia média": res["accs"].mean(),
        "Variância da acurácia": res["accs"].var(ddof=1),
        "Precisão (macro) média": res["prec_macros"].mean(),
        "Precisão (weighted) média": res["prec_weighteds"].mean(),
    }

def save_confusion_heatmap(conf_norm_df: pd.DataFrame, out_png: Path, title: str):
    """Gera um heatmap simples (matplotlib puro) da matriz de confusão normalizada."""
    plt.figure()
    mat = conf_norm_df.values
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.xticks(range(mat.shape[1]), conf_norm_df.columns)
    plt.yticks(range(mat.shape[0]), conf_norm_df.index)
    # Escrever valores na célula
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

# -------------------------------
# Pipeline principal
# -------------------------------
def main():
    args = build_argparser().parse_args()

    script_dir = Path(__file__).resolve().parent
    data_path = (script_dir / args.data).resolve()
    names_path = (script_dir / args.names).resolve()
    out_dir = (script_dir / args.out_dir).resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) Carregar dados
    df = load_dermatology_csv(data_path)

    # 2) Inferir coluna de classe e separar X/y
    cls_col = infer_class_column(df)
    X = df.drop(columns=[cls_col]).copy()
    y = df[cls_col].copy()

    # 3) Numéricos + imputação de NaN (ex.: idade '?')
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    # 4) Nomes de atributos
    feature_names = try_parse_feature_names(names_path, X.shape[1])
    X.columns = feature_names

    # 5) Grupos de atributos
    clin_cols, hist_cols = split_feature_groups(feature_names)

    # 6) Experimentos
    res_all  = run_experiment(X,               y, n_runs=args.n_runs, test_size=args.test_size, seed_base=args.seed)
    res_clin = run_experiment(X[clin_cols],    y, n_runs=args.n_runs, test_size=args.test_size, seed_base=args.seed)
    res_hist = run_experiment(X[hist_cols],   y, n_runs=args.n_runs, test_size=args.test_size, seed_base=args.seed)

    # 7) Resumo em DataFrame
    summary_df = pd.DataFrame([
        summarize("Todos os atributos",        res_all),
        summarize("Apenas clínicos",           res_clin),
        summarize("Apenas histopatológicos",   res_hist),
    ])
    summary_df.to_csv(out_dir / "resumo_metricas.csv", index=False)

    # 8) Matrizes de confusão (todos os atributos)
    conf_sum = res_all["confusion_sum"]
    conf_norm = conf_sum / conf_sum.sum(axis=1, keepdims=True)
    conf_df = pd.DataFrame(conf_sum, index=res_all["classes"], columns=res_all["classes"])
    conf_norm_df = pd.DataFrame(np.round(conf_norm, 3), index=res_all["classes"], columns=res_all["classes"])
    conf_df.to_csv(out_dir / "matriz_confusao_soma_50.csv")
    conf_norm_df.to_csv(out_dir / "matriz_confusao_normalizada.csv")

    # 9) Importância dos atributos (todos os atributos)
    feat_imp = res_all["feat_importance"]
    feat_imp_df = pd.DataFrame({
        "atributo": feature_names,
        "importancia_media": feat_imp
    }).sort_values("importancia_media", ascending=False).reset_index(drop=True)
    feat_imp_df.to_csv(out_dir / "importancia_media_atributos.csv", index=False)

    # 10) Gráficos
    # 10.1 Importâncias (Top-k)
    topk = min(15, len(feat_imp_df))
    plt.figure()
    plt.bar(range(topk), feat_imp_df["importancia_media"][:topk].values)
    plt.xticks(range(topk), feat_imp_df["atributo"][:topk].values, rotation=45, ha="right")
    plt.title(f"Top {topk} atributos por importância média (Perceptron OvR)")
    plt.tight_layout()
    plt.savefig(out_dir / "top_importancias.png", dpi=140)
    plt.close()

    # 10.2 Heatmap de confusão normalizada (opcional)
    if args.plot_heatmap:
        save_confusion_heatmap(conf_norm_df, out_dir / "matriz_confusao_normalizada_heatmap.png",
                               "Matriz de confusão normalizada (todos os atributos)")

    # 11) Prints úteis no console
    print("\n==== RESUMO –", args.n_runs, "rodadas ====")
    print(summary_df.to_string(index=False))
    print("\nArquivos gerados em:", out_dir)
    for fn in [
        "resumo_metricas.csv",
        "matriz_confusao_soma_50.csv",
        "matriz_confusao_normalizada.csv",
        "importancia_media_atributos.csv",
        "top_importancias.png",
        "matriz_confusao_normalizada_heatmap.png" if args.plot_heatmap else None,
    ]:
        if fn:
            print(" -", fn)

if __name__ == "__main__":
    main()
