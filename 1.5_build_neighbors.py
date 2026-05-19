"""
1.5 — Enriquecimento com medições dos vizinhos mais próximos.

Para cada variável gerada pelo 1.4_clean_data.py, constrói um DataFrame
onde cada registro da estação-alvo é enriquecido com as medições e as
informações espaciais das K estações mais próximas disponíveis no mesmo
timestamp.

Schema de saída (um arquivo por variável):

    code           — estação-alvo
    time           — timestamp da medição (UTC)
    measurement    — valor da estação-alvo
    n01 … n20      — medição da k-ésima estação vizinha disponível
    d01 … d20      — distance_km entre alvo e esse vizinho
    a01 … a20      — delta_altitude_m entre alvo e esse vizinho
    b01_sin…b20_sin — sin(azimute) do vizinho em relação ao alvo
    b01_cos…b20_cos — cos(azimute) do vizinho em relação ao alvo

Os slots variam por linha porque o vizinho que ocupa o slot k muda
conforme a disponibilidade de dados naquele timestamp.
NaN em n_k implica NaN em d_k, a_k, b_k_sin e b_k_cos.

"Disponível" significa: a estação-vizinha possui valor não-nulo para
aquela variável naquele timestamp exato.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITMO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Pivot: transforma cada variável numa matriz (time × station).
   Células NaN onde a estação não tem dado naquele timestamp.

2. Para cada estação-alvo, carrega os arrays estáticos de distância e
   altitude na ordem de vizinhança pré-computada.

3. Aplica first_k_valid_multi: única passagem pela máscara de NaN que
   seleciona os K primeiros válidos simultaneamente em três matrizes
   (valores, distâncias, altitudes), garantindo consistência de slots.

4. Grava incrementalmente com PyArrow para não acumular em memória.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
    data/temperature.parquet
    data/humidity.parquet
    data/rainfall.parquet
    data/global_radiation.parquet
    data/pressure.parquet
    data/station_distances.parquet

Output:
    data/temperature_neighbors.parquet
    data/humidity_neighbors.parquet
    data/rainfall_neighbors.parquet
    data/global_radiation_neighbors.parquet
    data/pressure_neighbors.parquet

Usage:
    python 1.5_build_neighbors.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DATA_DIR       = Path(__file__).parent / "data"
DISTANCES_PATH = DATA_DIR / "station_distances.parquet"
STATIONS_PATH  = DATA_DIR / "stations.parquet"

K = 20  # vizinhos a reter

# Filtro de data — defina None para processar o dataset completo
DATE_START = None   # ex: "2025-03-25"
DATE_END   = None   # ex: "2025-03-25"

VARIABLES = [
    "temperature",
    "humidity",
    "rainfall",
    "global_radiation",
    "pressure",
]

# Defina None para rodar todas as variáveis, ou uma lista para filtrar
VARIABLES_TO_RUN = ["temperature"]  # ou None

# Todas as variáveis são arredondadas para 2 casas decimais
ROUND_2 = set(VARIABLES)


# ── schema PyArrow ────────────────────────────────────────────────────────────

OUTPUT_SCHEMA = pa.schema([
    ("code",        pa.string()),
    ("time",        pa.timestamp("us")),
    ("measurement", pa.float32()),
    *[(f"n{i+1:02d}",     pa.float32()) for i in range(K)],
    *[(f"d{i+1:02d}",     pa.float32()) for i in range(K)],
    *[(f"a{i+1:02d}",     pa.float32()) for i in range(K)],
    *[(f"b{i+1:02d}_sin", pa.float32()) for i in range(K)],
    *[(f"b{i+1:02d}_cos", pa.float32()) for i in range(K)],
    ("hour_sin",    pa.float32()),
    ("hour_cos",    pa.float32()),
    ("doy_sin",     pa.float32()),
    ("doy_cos",     pa.float32()),
])


# ── utilidades ────────────────────────────────────────────────────────────────

DistanceIndex = dict[str, tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def _bearing_sin_cos(
    lat1: float, lon1: float,
    lats2: np.ndarray, lons2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized azimute (sin, cos) de um ponto de origem para N destinos."""
    lat1_r  = np.radians(lat1)
    lon1_r  = np.radians(lon1)
    lats2_r = np.radians(lats2)
    lons2_r = np.radians(lons2)
    dlon = lons2_r - lon1_r
    x = np.sin(dlon) * np.cos(lats2_r)
    y = np.cos(lat1_r) * np.sin(lats2_r) - np.sin(lat1_r) * np.cos(lats2_r) * np.cos(dlon)
    b = np.arctan2(x, y)  # radianos, -π a π
    return np.sin(b).astype(np.float32), np.cos(b).astype(np.float32)


def load_distance_index() -> DistanceIndex:
    """
    Retorna {from_code: (to_codes, distance_km_arr, delta_altitude_arr,
                         bearing_sin_arr, bearing_cos_arr)}
    ordenados por effective_distance_km (ordem já garantida pelo 1.2).
    """
    dist     = pd.read_parquet(DISTANCES_PATH).reset_index()
    stations = (
        pd.read_parquet(STATIONS_PATH, columns=["code", "latitude", "longitude"])
        .set_index("code")
        .astype({"latitude": np.float64, "longitude": np.float64})
    )

    index: DistanceIndex = {}
    for from_code, group in dist.groupby("from_code"):
        if from_code not in stations.index:
            continue
        lat1 = stations.at[from_code, "latitude"]
        lon1 = stations.at[from_code, "longitude"]
        to_codes = group["to_code"].tolist()
        lats2 = stations.loc[group["to_code"].values, "latitude"].values
        lons2 = stations.loc[group["to_code"].values, "longitude"].values
        bsin, bcos = _bearing_sin_cos(lat1, lon1, lats2, lons2)
        index[from_code] = (
            to_codes,
            group["distance_km"].values.astype(np.float32),
            group["delta_altitude_m"].values.astype(np.float32),
            bsin,
            bcos,
        )
    return index


def first_k_valid_multi(
    val_arr:  np.ndarray,
    dist_arr: np.ndarray,
    dalt_arr: np.ndarray,
    bsin_arr: np.ndarray,
    bcos_arr: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Para cada linha seleciona os k primeiros slots não-NaN (baseado em
    val_arr) e aplica a mesma seleção em todos os arrays estáticos 1-D.

    Entradas : val_arr (n_rows, n_cols); demais são 1-D estáticos por estação.
    Saídas   : cinco arrays (n_rows, k) com NaN nos slots não preenchidos.
    """
    mask     = ~np.isnan(val_arr)
    cumcount = np.cumsum(mask, axis=1)
    select   = mask & (cumcount <= k)
    rows, cols = np.where(select)
    positions  = cumcount[rows, cols] - 1   # índice 0-based na saída

    n_rows = val_arr.shape[0]
    top_val  = np.full((n_rows, k), np.nan, dtype=np.float32)
    top_dist = np.full((n_rows, k), np.nan, dtype=np.float32)
    top_dalt = np.full((n_rows, k), np.nan, dtype=np.float32)
    top_bsin = np.full((n_rows, k), np.nan, dtype=np.float32)
    top_bcos = np.full((n_rows, k), np.nan, dtype=np.float32)

    top_val[rows, positions]  = val_arr[rows, cols]
    top_dist[rows, positions] = dist_arr[cols]
    top_dalt[rows, positions] = dalt_arr[cols]
    top_bsin[rows, positions] = bsin_arr[cols]
    top_bcos[rows, positions] = bcos_arr[cols]

    return top_val, top_dist, top_dalt, top_bsin, top_bcos


# ── processamento por variável ────────────────────────────────────────────────

def process_variable(
    variable:     str,
    dist_index:   DistanceIndex,
    start:        str | None = None,
    end:          str | None = None,
) -> None:
    input_path  = DATA_DIR / f"{variable}.parquet"
    suffix      = f"_{start}_{end}" if (start or end) else ""
    output_path = DATA_DIR / f"{variable}_neighbors{suffix}.parquet"

    if not input_path.exists():
        print(f"  SKIP {variable}: {input_path} não encontrado — rode 1.4_clean_data.py primeiro.")
        return

    round_measurement = variable in ROUND_2
    t0 = time.time()
    print(f"\n[{variable}]")

    # 1. Pivot → matrix (time × station)
    df = pd.read_parquet(input_path)
    if start:
        df = df[df["time"] >= start]
    if end:
        df = df[df["time"] <= end]
    if df.empty:
        print("  Nenhum dado no intervalo especificado.")
        return
    pivot = df.pivot(index="time", columns="code", values="measurement").sort_index()
    del df

    pivot_codes = pivot.columns.tolist()
    code_to_idx = {c: i for i, c in enumerate(pivot_codes)}
    pivot_arr   = pivot.values.astype(np.float32)
    time_index  = pivot.index
    print(f"  Pivot: {pivot_arr.shape[0]:,} timestamps × {pivot_arr.shape[1]} estações")

    # 2. Gravar estação a estação
    n_written = 0
    with pq.ParquetWriter(str(output_path), OUTPUT_SCHEMA, compression="snappy") as writer:
        for target_code in tqdm(pivot_codes, desc="  estações", unit="stn"):
            if target_code not in dist_index:
                continue

            all_codes, all_dist, all_dalt, all_bsin, all_bcos = dist_index[target_code]

            # filtra vizinhos presentes nesta variável, mantendo a ordem
            mask_present = [c in code_to_idx for c in all_codes]
            present_idx  = [i for i, ok in enumerate(mask_present) if ok]
            ordered  = [all_codes[i] for i in present_idx]
            dist_row = all_dist[present_idx]
            dalt_row = all_dalt[present_idx]
            bsin_row = all_bsin[present_idx]
            bcos_row = all_bcos[present_idx]

            if not ordered:
                continue

            nbr_idx = np.array([code_to_idx[c] for c in ordered], dtype=np.int32)

            # linhas onde o alvo tem dado
            tgt_col   = pivot_arr[:, code_to_idx[target_code]]
            valid_row = ~np.isnan(tgt_col)
            if not valid_row.any():
                continue

            valid_times  = time_index[valid_row]
            valid_target = tgt_col[valid_row]
            if round_measurement:
                valid_target = np.round(valid_target, 2)

            # sub-matrix de vizinhos
            nbr_matrix = pivot_arr[np.ix_(valid_row, nbr_idx)]

            # primeiros K não-NaN + features espaciais correspondentes
            top_val, top_dist, top_dalt, top_bsin, top_bcos = first_k_valid_multi(
                nbr_matrix, dist_row, dalt_row, bsin_row, bcos_row, K
            )

            n = len(valid_times)
            hours = valid_times.hour.to_numpy().astype(np.float32)
            doys  = valid_times.day_of_year.to_numpy().astype(np.float32)
            table = pa.table(
                {
                    "code":        pa.array([target_code] * n, type=pa.string()),
                    "time":        pa.array(valid_times.to_numpy(), type=pa.timestamp("us")),
                    "measurement": pa.array(valid_target,           type=pa.float32()),
                    **{f"n{i+1:02d}":     pa.array(top_val[:, i],  type=pa.float32()) for i in range(K)},
                    **{f"d{i+1:02d}":     pa.array(top_dist[:, i], type=pa.float32()) for i in range(K)},
                    **{f"a{i+1:02d}":     pa.array(top_dalt[:, i], type=pa.float32()) for i in range(K)},
                    **{f"b{i+1:02d}_sin": pa.array(top_bsin[:, i], type=pa.float32()) for i in range(K)},
                    **{f"b{i+1:02d}_cos": pa.array(top_bcos[:, i], type=pa.float32()) for i in range(K)},
                    "hour_sin": pa.array(np.sin(2 * np.pi * hours / 24),  type=pa.float32()),
                    "hour_cos": pa.array(np.cos(2 * np.pi * hours / 24),  type=pa.float32()),
                    "doy_sin":  pa.array(np.sin(2 * np.pi * doys  / 365), type=pa.float32()),
                    "doy_cos":  pa.array(np.cos(2 * np.pi * doys  / 365), type=pa.float32()),
                },
                schema=OUTPUT_SCHEMA,
            )
            writer.write_table(table)
            n_written += n

    size_mb = output_path.stat().st_size / 1_048_576
    elapsed = time.time() - t0
    print(f"  → {output_path.name}   {n_written:,} registros   {size_mb:.0f} MB   {elapsed:.0f}s")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== 1.5 Build Neighbors ===")
    if DATE_START or DATE_END:
        print(f"  Datas    : {DATE_START or 'início'} → {DATE_END or 'fim'}")
    if VARIABLES_TO_RUN is not None:
        print(f"  Variáveis: {VARIABLES_TO_RUN}")
    print()

    if not DISTANCES_PATH.exists():
        print(f"ERROR: {DISTANCES_PATH} não encontrado. Execute 1.2_compute_station_distances.py primeiro.")
        sys.exit(1)

    print("Carregando índice de distâncias...")
    dist_index = load_distance_index()
    print(f"Estações indexadas: {len(dist_index):,}")

    variables = VARIABLES_TO_RUN if VARIABLES_TO_RUN is not None else VARIABLES
    for variable in variables:
        process_variable(variable, dist_index, start=DATE_START, end=DATE_END)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
