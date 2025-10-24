import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

XLSX = "resultados_pli_e_meta.xlsx"
CSV_MAIN = "main.csv"
CSV_GRASP = "grasp_results.csv"
CSV_TABU = "tabu_results.csv"
CSV_GA = "ga_results.csv"
OUTDIR = "pp_out"
QUALITY_LEVELS = [1.00, 1.01, 1.05, 1.10, 1.20]
EPS = 1e-12
GA_NOMES_PREFERIDOS = [
    "POP_MUT_EVOL1",
    "POP+MUT+EVOL1",
    "POP-MUT-EVOL1",
    "POP MUT EVOL1",
    "Pop_Mut_Evol1",
    "Pop+Mut+Evol1",
    "Pop-Mut-Evol1",
    "Pop Mut Evol1",
    "PopMutEvol1",
]
GA_CONJUNTOS_TOKENS = [["pop", "mut", "evol1"], ["pop", "mut", "evol"]]


def normalizar_instancia(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if s.endswith(".txt"):
        s = s[:-4]
    return s


def manter_padrao_15(df):
    return df[df["instancia"].str.match(r"^n\d+p\d+$", na=False)].copy()


def ler_excel(caminho, aba):
    try:
        return pd.read_excel(caminho, sheet_name=aba)
    except Exception:
        return None


def deduzir_header(df):
    if df is None or df.empty:
        return df
    primeira = df.iloc[0].astype(str).str.lower().tolist()
    if any("inst" in c for c in primeira) or any("config" in c for c in primeira):
        head = df.iloc[0].astype(str).tolist()
        df = df.iloc[1:].copy()
        cont, cols = {}, []
        for c in head:
            if c not in cont:
                cont[c] = 0
                cols.append(c)
            else:
                cont[c] += 1
                cols.append(f"{c}.{cont[c]}")
        df.columns = [str(c).strip() for c in cols]
    return df


def carregar_main(caminho_xlsx, caminho_csv):
    df = ler_excel(caminho_xlsx, "main")
    if df is None and os.path.exists(caminho_csv):
        df = pd.read_csv(caminho_csv)
    df = deduzir_header(df)
    if df is None:
        return pd.DataFrame(columns=["instancia", "PLI_value"])
    ren = {}
    for c in ["Instância", "Instancia", "instancia", "INSTÂNCIA", "INSTANCIA"]:
        if c in df.columns:
            ren[c] = "instancia"
            break
    df = df.rename(columns=ren)
    if "PLI_value" not in df.columns and "LI" in df.columns:
        df = df.rename(columns={"LI": "PLI_value"})
    if "instancia" not in df.columns:
        df["instancia"] = ""
    df["instancia"] = df["instancia"].astype(str).map(normalizar_instancia)
    df = manter_padrao_15(df)[["instancia", "PLI_value"]].reset_index(drop=True)
    return df


def padronizar_metodo(df, mapa):
    df = df.rename(columns=mapa)
    df["instancia"] = df["instancia"].astype(str).map(normalizar_instancia)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = (
        manter_padrao_15(df)
        .dropna(subset=["instancia", "config", "value"])
        .reset_index(drop=True)
    )
    return df


def carregar_metodo(caminho_xlsx, aba, caminho_csv, mapa):
    df = ler_excel(caminho_xlsx, aba)
    if df is None and os.path.exists(caminho_csv):
        df = pd.read_csv(caminho_csv)
    return padronizar_metodo(df, mapa)


def melhor_por_config(df):
    return df.groupby(["instancia", "config"], as_index=False).agg(
        best_value=("value", "max"), best_time=("time", "min")
    )


def melhor_global(grasp_cfg, tabu_cfg, ga_cfg):
    cat = pd.concat(
        [
            grasp_cfg[["instancia", "best_value"]],
            tabu_cfg[["instancia", "best_value"]],
            ga_cfg[["instancia", "best_value"]],
        ],
        ignore_index=True,
    )
    return (
        cat.groupby("instancia", as_index=False)["best_value"]
        .max()
        .rename(columns={"best_value": "B_i"})
    )


def escolher_config(df_cfg, base_B):
    d = df_cfg.merge(base_B, on="instancia", how="right")
    linhas = []
    for cfg, g in d.groupby("config"):
        r = []
        for _, row in g.iterrows():
            Bi, vi = row["B_i"], row["best_value"]
            if pd.isna(Bi) or pd.isna(vi):
                r.append(np.inf)
            else:
                r.append(1.0 + (Bi - vi) / max(EPS, abs(Bi)))
        r = np.array(r)
        linhas.append(
            {
                "config": cfg,
                "rho@1.00": float((r <= 1.00).mean()),
                "rho@1.01": float((r <= 1.01).mean()),
                "rho@1.05": float((r <= 1.05).mean()),
                "rho@1.10": float((r <= 1.10).mean()),
                "rho@1.20": float((r <= 1.20).mean()),
                "_mean_r": float(np.nanmean(np.where(np.isfinite(r), r, np.inf))),
            }
        )
    resumo = (
        pd.DataFrame(linhas)
        .sort_values(
            ["rho@1.00", "rho@1.01", "rho@1.05", "rho@1.10", "rho@1.20", "_mean_r"],
            ascending=[False, False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    nome = str(resumo.loc[0, "config"])
    vencedor = df_cfg[df_cfg["config"] == nome].copy()
    return nome, vencedor, resumo


def normalizar_nome_cfg(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def filtrar_ga_preferido(df_cfg):
    cfgs = df_cfg["config"].astype(str)
    nomes = {n.lower().strip() for n in GA_NOMES_PREFERIDOS}
    m1 = cfgs.str.lower().str.strip().isin(nomes)
    if m1.any():
        return df_cfg[m1]
    m2 = pd.Series(False, index=df_cfg.index)
    for n in nomes:
        if n:
            m2 = m2 | cfgs.str.lower().str.contains(re.escape(n), na=False)
    if m2.any():
        return df_cfg[m2]
    norm = cfgs.map(normalizar_nome_cfg)
    for toks in GA_CONJUNTOS_TOKENS:
        toks = [t.strip().lower() for t in toks if t.strip()]
        m3 = norm.apply(lambda s: all(t in s for t in toks))
        sub = df_cfg[m3]
        if not sub.empty:
            return sub
    pool = sorted({t for ts in GA_CONJUNTOS_TOKENS for t in ts})

    def score(s):
        s = normalizar_nome_cfg(s)
        return sum(1 for t in pool if t in s)

    pont = cfgs.apply(score)
    if pont.max() > 0:
        return df_cfg.loc[[pont.idxmax()]]
    return df_cfg.iloc[0:0]


def montar_tabela_final(grasp_venc, tabu_venc, ga_venc, main_df):
    insts = sorted(
        set(grasp_venc["instancia"])
        | set(tabu_venc["instancia"])
        | set(ga_venc["instancia"])
    )
    final = pd.DataFrame({"instancia": insts})
    final = final.merge(
        grasp_venc[["instancia", "best_value"]].rename(
            columns={"best_value": "GRASP_value"}
        ),
        on="instancia",
        how="left",
    )
    final = final.merge(
        tabu_venc[["instancia", "best_value"]].rename(
            columns={"best_value": "TABU_value"}
        ),
        on="instancia",
        how="left",
    )
    final = final.merge(
        ga_venc[["instancia", "best_value"]].rename(columns={"best_value": "GA_value"}),
        on="instancia",
        how="left",
    )
    if "PLI_value" in main_df.columns:
        final = final.merge(
            main_df[["instancia", "PLI_value"]], on="instancia", how="left"
        )
    return manter_padrao_15(final).reset_index(drop=True)


def calcular_razoes(df_sub, metodos):
    insts = df_sub["instancia"].tolist()
    M = df_sub[[f"{m}_value" for m in metodos]].values.astype(float)
    Bv = np.nanmax(M, axis=1)
    linhas = []
    for i, inst in enumerate(insts):
        Bi = Bv[i]
        denom = max(EPS, abs(Bi)) if not np.isnan(Bi) else EPS
        for m in metodos + (["PLI"] if "PLI_value" in df_sub.columns else []):
            vi = (
                df_sub.loc[i, "PLI_value"]
                if m == "PLI"
                else df_sub.loc[i, f"{m}_value"]
            )
            vi = float(vi) if pd.notna(vi) else np.nan
            r = np.inf if (np.isnan(Bi) or np.isnan(vi)) else 1.0 + (Bi - vi) / denom
            linhas.append({"instancia": inst, "metodo": m, "r": float(r)})
    return pd.DataFrame(linhas)


def gerar_curvas(df_ratios, taus):
    insts = sorted(df_ratios["instancia"].unique())
    n = float(len(insts)) if insts else 1.0
    blocos = []
    for m, g in df_ratios.groupby("metodo"):
        r = g["r"].values
        rho = [(r <= t).sum() / n for t in taus]
        blocos.append(pd.DataFrame({"tau": taus, "rho": rho, "metodo": m}))
    return pd.concat(blocos, ignore_index=True)


def salvar_csv(caminho, df):
    caminho.write_text(df.to_csv(index=False), encoding="utf-8")


def plotar_curvas(df_curvas, titulo, png, xlim=None):
    plt.figure(figsize=(6.6, 4.8))
    plt.ylim(0.0, 1.03)
    for m in ["PLI", "GA", "TABU", "GRASP"]:
        if m not in df_curvas["metodo"].unique():
            continue
        g = df_curvas[df_curvas["metodo"] == m].sort_values("tau")
        plt.step(g["tau"].values, g["rho"].values, where="post", label=m)
    if xlim is not None:
        plt.xlim(*xlim)
        if xlim[0] >= 1.0 and xlim[1] <= 2.0:
            ticks = [
                t for t in [1.00, 1.01, 1.05, 1.10, 1.20] if xlim[0] <= t <= xlim[1]
            ]
            if ticks:
                plt.xticks(ticks, [f"{t:.2f}" for t in ticks])
    plt.grid(True, alpha=0.35)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\rho(\tau)$")
    plt.title(titulo)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()


def resumir_niveis(df_ratios, niveis):
    insts = sorted(df_ratios["instancia"].unique())
    n = float(len(insts)) if insts else 1.0
    linhas = []
    for m, g in df_ratios.groupby("metodo"):
        r = g["r"].values
        linha = {"metodo": m}
        for lv in niveis:
            linha[f"rho@{lv:.2f}"] = float((r <= lv).sum()) / n
        linhas.append(linha)
    return pd.DataFrame(linhas)


def gerar_saidas_por_subconjunto(tag, df_sub, metodos, pasta):
    df_sub = df_sub.reset_index(drop=True)
    pasta.mkdir(parents=True, exist_ok=True)
    ratios = calcular_razoes(df_sub, metodos)
    salvar_csv(pasta / f"quality_ratios_{tag}.csv", ratios)
    fin = ratios[np.isfinite(ratios["r"])]
    tau_max = max(1.02, float(fin["r"].max())) if not fin.empty else 1.02
    taus_full = np.linspace(1.0, tau_max, 200)
    curvas_full = gerar_curvas(ratios, taus_full)
    salvar_csv(pasta / f"quality_curves_full_{tag}.csv", curvas_full)
    plotar_curvas(
        curvas_full,
        f"Performance Profile ({tag})",
        pasta / f"quality_full_{tag}.png",
        xlim=(1.0, tau_max),
    )
    for perc in [1.01, 1.05, 1.10, 1.20]:
        taus_zoom = np.linspace(1.0, perc, 200)
        curvas_zoom = gerar_curvas(ratios, taus_zoom)
        salvar_csv(
            pasta / f"quality_curves_zoom{int((perc-1)*100):02d}_{tag}.csv", curvas_zoom
        )
        plotar_curvas(
            curvas_zoom,
            f"Performance Profile ({tag}) — até {int((perc-1)*100)}%",
            pasta / f"quality_zoom{int((perc-1)*100):02d}_{tag}.png",
            xlim=(1.0, perc),
        )
    resumo = resumir_niveis(ratios, QUALITY_LEVELS)
    salvar_csv(pasta / f"quality_summary_{tag}.csv", resumo)


def main():
    aqui = Path(__file__).resolve().parent
    outdir = aqui / OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    main_df = carregar_main(aqui / XLSX, aqui / CSV_MAIN)
    grasp_raw = carregar_metodo(
        aqui / XLSX,
        "grasp_results",
        aqui / CSV_GRASP,
        {
            "Instância": "instancia",
            "Configuração": "config",
            "MS": "value",
            "Tempo (s)": "time",
        },
    )
    tabu_raw = carregar_metodo(
        aqui / XLSX,
        "tabu_results",
        aqui / CSV_TABU,
        {
            "Instance": "instancia",
            "Configuration": "config",
            "BestValue": "value",
            "ExecutionTime(s)": "time",
        },
    )
    ga_raw = carregar_metodo(
        aqui / XLSX,
        "ga_results",
        aqui / CSV_GA,
        {
            "Instance": "instancia",
            "Configuration": "config",
            "BestSolution": "value",
            "Time_s": "time",
        },
    )
    grasp_cfg = melhor_por_config(grasp_raw)
    tabu_cfg = melhor_por_config(tabu_raw)
    ga_cfg = melhor_por_config(ga_raw)
    B = melhor_global(grasp_cfg, tabu_cfg, ga_cfg)
    nome_grasp, grasp_venc, _ = escolher_config(grasp_cfg, B)
    nome_tabu, tabu_venc, _ = escolher_config(tabu_cfg, B)
    ga_pref = filtrar_ga_preferido(ga_cfg)
    if ga_pref.empty:
        nome_ga, ga_venc, _ = escolher_config(ga_cfg, B)
    else:
        nome_ga, ga_venc, _ = escolher_config(ga_pref, B)
    final = montar_tabela_final(grasp_venc, tabu_venc, ga_venc, main_df)
    salvar_csv(outdir / "tabela_final_valores.csv", final)
    metodos = ["GRASP", "TABU", "GA"]
    subconjuntos = {
        "Todas_Instâncias": final.copy(),
        "p1": final[final["instancia"].str.endswith("p1")].copy(),
        "p2": final[final["instancia"].str.endswith("p2")].copy(),
        "p3": final[final["instancia"].str.endswith("p3")].copy(),
    }
    for tag, df_sub in subconjuntos.items():
        gerar_saidas_por_subconjunto(tag, df_sub, metodos, outdir / tag)
    print("Configs escolhidas:")
    print("  GRASP:", nome_grasp)
    print("  TABU :", nome_tabu)
    print("  GA   :", nome_ga)
    print(f"Saídas em: {OUTDIR}")


if __name__ == "__main__":
    main()
