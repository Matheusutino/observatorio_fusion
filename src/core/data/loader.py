"""
Funções de carregamento de dados e geração/cache de embeddings.
"""
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..config.config import (
    DATA_PATH,
    EMBEDDINGS_DIR,
    SBERT_MODEL,
)


def load_data() -> pd.DataFrame:
    """
    Carrega o dataset a partir do parquet, realiza preprocessamento básico
    e cria a variável alvo 'nivel_binario' (ALTA/BAIXA).
    """
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=["modelo_nivel", "descricao_resumo", "tema"])
    df = df[df["descricao_resumo"].str.strip() != ""].reset_index(drop=True)
    df["nivel_binario"] = df["modelo_nivel"].str.upper().map(
        lambda x: "ALTA" if x == "ALTA" else "BAIXA"
    )
    print(f"Shape: {df.shape}")
    print(df["nivel_binario"].value_counts())
    return df


def _kw_to_str(series: pd.Series) -> pd.Series:
    """Converte lista de keywords em string concatenada."""
    return series.apply(
        lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else (x or "")
    )


def get_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna embeddings da configuração padrão (titulo_resumo_abstract, nome_keywords).
    """
    return get_embeddings_for_config(df, "titulo_resumo_abstract", "nome_keywords")


def get_embeddings_for_config(
    df: pd.DataFrame, prod_cfg: str, tema_cfg: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega ou gera embeddings para uma configuração de fonte específica.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    prod_cfg : str
        Configuração da produção: "titulo" | "resumo" | "abstract" | "keywords" | "resumo_abstract" | "titulo_resumo_abstract"
    tema_cfg : str
        Configuração do tema: "nome" | "nome_keywords"

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        (emb_prod, emb_tema) - embeddings normalizados em float32.
    """
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    cache_path = os.path.join(
        EMBEDDINGS_DIR, f"emb_prod={prod_cfg}_tema={tema_cfg}.npz"
    )

    if os.path.exists(cache_path):
        print(f"  Cache: {cache_path}")
        data = np.load(cache_path)
        return data["emb_prod"].astype(np.float32), data["emb_tema"].astype(
            np.float32
        )

    print(f"  Gerando embeddings [{prod_cfg} | {tema_cfg}]...")
    sbert = SentenceTransformer(SBERT_MODEL)

    kw_prod = _kw_to_str(df["descricao_palavra_chave"])
    prod_map = {
        "titulo": df["nome_producao"].fillna(""),
        "resumo": df["descricao_resumo"].fillna(""),
        "abstract": df["descricao_abstract"].fillna(""),
        "keywords": kw_prod,
        "resumo_abstract": df["descricao_resumo"].fillna("")
        + " "
        + df["descricao_abstract"].fillna(""),
        "titulo_resumo_abstract": df["nome_producao"].fillna("")
        + " "
        + df["descricao_resumo"].fillna("")
        + " "
        + df["descricao_abstract"].fillna(""),
    }

    kw_tema = _kw_to_str(df["palavras_chave"])
    tema_map = {
        "nome": df["tema"].fillna(""),
        "nome_keywords": df["tema"].fillna("") + " " + kw_tema,
    }

    emb_prod = sbert.encode(
        prod_map[prod_cfg].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    emb_tema = sbert.encode(
        tema_map[tema_cfg].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.savez_compressed(cache_path, emb_prod=emb_prod, emb_tema=emb_tema)
    print(f"  Salvo: {cache_path}")
    return emb_prod, emb_tema
