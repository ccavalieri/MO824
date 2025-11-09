#!/usr/bin/env python3
"""
Risk Generator for School Dropout Problem
Uses INEP data and Markov chain simulation
"""

import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path


# Instance size configurations
SIZE_CONFIG = {
    "classroom": 40,
    "school": 1000,
    "neighborhood": 10000,
    "rural_city": 75000,
    "country_city": 150000,
    "urban_city": 250000,
    "state": 1000000
}


# Colunas do arquivo INEP
COLUNAS_INEP = ['Ano',
 'Região',
 'UF',
 'Código do município',
 'Nome do município',
 'Localização',
 'Dependência Administrativa',
 'Taxa de Promoção_Ensino Fundamental_Total',
 'Taxa de Promoção_Ensino Fundamental_Anos Iniciais',
 'Taxa de Promoção_Ensino Fundamental_Anos Finais',
 'Taxa de Promoção_Ensino Fundamental_1º Ano',
 'Taxa de Promoção_Ensino Fundamental_2º Ano',
 'Taxa de Promoção_Ensino Fundamental_3º Ano',
 'Taxa de Promoção_Ensino Fundamental_4º Ano',
 'Taxa de Promoção_Ensino Fundamental_5º Ano',
 'Taxa de Promoção_Ensino Fundamental_6º Ano',
 'Taxa de Promoção_Ensino Fundamental_7º Ano',
 'Taxa de Promoção_Ensino Fundamental_8º Ano',
 'Taxa de Promoção_Ensino Fundamental_9º Ano',
 'Taxa de Promoção_ Ensino Médio_Total',
 'Taxa de Promoção_ Ensino Médio_1ª série',
 'Taxa de Promoção_ Ensino Médio_2ª série',
 'Taxa de Promoção_ Ensino Médio_3ª série',
 'Taxa de Repetência_Ensino Fundamental_Total',
 'Taxa de Repetência_Ensino Fundamental_Anos Iniciais',
 'Taxa de Repetência_Ensino Fundamental_Anos Finais',
 'Taxa de Repetência_Ensino Fundamental_1º Ano',
 'Taxa de Repetência_Ensino Fundamental_2º Ano',
 'Taxa de Repetência_Ensino Fundamental_3º Ano',
 'Taxa de Repetência_Ensino Fundamental_4º Ano',
 'Taxa de Repetência_Ensino Fundamental_5º Ano',
 'Taxa de Repetência_Ensino Fundamental_6º Ano',
 'Taxa de Repetência_Ensino Fundamental_7º Ano',
 'Taxa de Repetência_Ensino Fundamental_8º Ano',
 'Taxa de Repetência_Ensino Fundamental_9º Ano',
 'Taxa de Repetência_ Ensino Médio_Total',
 'Taxa de Repetência_ Ensino Médio_1ª série',
 'Taxa de Repetência_ Ensino Médio_2ª série',
 'Taxa de Repetência_ Ensino Médio_3ª série',
 'Taxa de Evasão_Ensino Fundamental_Total',
 'Taxa de Evasão_Ensino Fundamental_Anos Iniciais',
 'Taxa de Evasão_Ensino Fundamental_Anos Finais',
 'Taxa de Evasão_Ensino Fundamental_1º Ano',
 'Taxa de Evasão_Ensino Fundamental_2º Ano',
 'Taxa de Evasão_Ensino Fundamental_3º Ano',
 'Taxa de Evasão_Ensino Fundamental_4º Ano',
 'Taxa de Evasão_Ensino Fundamental_5º Ano',
 'Taxa de Evasão_Ensino Fundamental_6º Ano',
 'Taxa de Evasão_Ensino Fundamental_7º Ano',
 'Taxa de Evasão_Ensino Fundamental_8º Ano',
 'Taxa de Evasão_Ensino Fundamental_9º Ano',
 'Taxa de Evasão_ Ensino Médio_Total',
 'Taxa de Evasão_ Ensino Médio_1ª série',
 'Taxa de Evasão_ Ensino Médio_2ª série',
 'Taxa de Evasão_ Ensino Médio_3ª série',
 'Migração para EJA_Ensino Fundamental_Total',
 'Migração para EJA_Ensino Fundamental_Anos Iniciais',
 'Migração para EJA_Ensino Fundamental_Anos Finais',
 'Migração para EJA_Ensino Fundamental_1º Ano',
 'Migração para EJA_Ensino Fundamental_2º Ano',
 'Migração para EJA_Ensino Fundamental_3º Ano',
 'Migração para EJA_Ensino Fundamental_4º Ano',
 'Migração para EJA_Ensino Fundamental_ 5º Ano',
 'Migração para EJA_Ensino Fundamental_6º Ano',
 'Migração para EJA_Ensino Fundamental_7º Ano',
 'Migração para EJA_Ensino Fundamental_8º Ano',
 'Migração para EJA_Ensino Fundamental_9º Ano',
 'Migração para EJA_ Ensino Médio_Total',
 'Migração para EJA_ Ensino Médio_1ª série',
 'Migração para EJA_ Ensino Médio_2ª série',
 'Migração para EJA_ Ensino Médio_3ª série']


def gerar_matriz_markov_inep(linha, complexidade=1, anos=9):
    """
    Gera matriz de Markov baseada em dados do INEP.
    """
    n_estados = anos + 3
    P = np.zeros((n_estados, n_estados))
    for i in range(anos):
        prom = linha[f'Taxa de Promoção_Ensino Fundamental_{i+1}º Ano'] / 100
        rep = linha[f'Taxa de Repetência_Ensino Fundamental_{i+1}º Ano'] / 100
        eva = linha[f'Taxa de Evasão_Ensino Fundamental_{i+1}º Ano'] / 100
        eja = linha.get(f'Migração para EJA_Ensino Fundamental_{i+1}º Ano', 0) / 100


        if complexidade >= 3:
            prom += np.random.normal(0, 0.01)
            rep += np.random.normal(0, 0.01)
            eva += np.random.normal(0, 0.01)
            eja += np.random.normal(0, 0.01)
            prom, rep, eva, eja = [max(0, min(1, x)) for x in [prom, rep, eva, eja]]

        total = prom + rep + eva + eja
        if total > 1:
            prom /= total
            rep /= total
            eva /= total
            eja /= total

        if i < anos - 1:
            P[i, i+1] = prom
        else:
            P[i, n_estados-1] = prom
        P[i, i] = rep
        P[i, anos+1] = eva
        P[i, anos] = eja


        if complexidade >= 4 and np.random.rand() < 0.05:
            if i < anos - 2:
                P[i, i+2] += 0.02
                P[i, i+1] -= 0.02

    for idx in [anos, anos+1, n_estados-1]:
        P[idx, idx] = 1.0

    return P


def generate_risk_profiles(n_students, linha_inep, complexidade_markov=3, seed=None):
    """
    Gera perfis de risco usando simulação de Markov.
    """
    if seed is not None:
        np.random.seed(seed)

    P = gerar_matriz_markov_inep(linha_inep, complexidade=complexidade_markov)
    anos = 9
    estados = [f"{i+1}º Ano" for i in range(anos)] + ["EJA", "Evasão", "Concluído"]

    anos_iniciais = np.random.choice(range(anos), size=n_students)
    estados_finais = []
    for ano in anos_iniciais:
        probs = P[ano]
        probs = np.maximum(probs, 0)
        soma = probs.sum()
        if soma == 0:
            probs = np.zeros_like(probs)
            probs[ano] = 1.0
        else:
            probs = probs / soma
        estado_final = np.random.choice(len(estados), p=probs)
        estados_finais.append(estado_final)
    riscos = []
    for estado in estados_finais:
        if estados[estado] == "Evasão":
            riscos.append(np.round(np.random.uniform(0.7, 0.95), 3))
        elif estados[estado] == "Concluído":
            riscos.append(np.round(np.random.uniform(0.05, 0.2), 3))
        elif estados[estado] == "EJA":
            riscos.append(np.round(np.random.uniform(0.4, 0.7), 3))
        else:
            riscos.append(np.round(np.random.uniform(0.2, 0.6), 3))

    return riscos


def load_inep_data():
    """
    Carrega dados do INEP.
    """
    # Tenta carregar de arquivos locais comuns
    file_name = 'transicao_escolas_cidades_2022_2021.xlsx'
    
    df = pd.read_excel(file_name)
    df.columns = COLUNAS_INEP
    return df


def save_risks(risks, output_file):
    """Save risks to a text file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(f"# School Dropout Problem - Initial Risks\n")
        f.write(f"# Number of students: {len(risks)}\n")
        f.write(f"# Format: student_id risk_value\n")
        f.write(f"#\n")
        
        for idx, risk in enumerate(risks):
            f.write(f"{idx} {risk:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate initial dropout risks for students",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--size",
        type=str,
        required=True,
        choices=SIZE_CONFIG.keys()
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None
    )
    
    args = parser.parse_args()
    
    # Determine output filename
    if args.output is None:
        args.output = f"risks_{args.size}.txt"
    
    # Generate risks
    n_students = SIZE_CONFIG[args.size]
    print(f"Generating risks for {n_students:,} students ({args.size})")
    
    # Carrega dados INEP
    df_inep = load_inep_data()
    
    # Seleciona uma linha
    linha_inep = df_inep[df_inep['Dependência Administrativa'] == 'Pública'].sample(1).iloc[0].to_dict()
    
    # Gera riscos usando Markov
    risks = generate_risk_profiles(
        n_students, 
        linha_inep, 
        complexidade_markov=3, 
        seed=args.seed
    )
    
    # Save to file
    save_risks(risks, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
