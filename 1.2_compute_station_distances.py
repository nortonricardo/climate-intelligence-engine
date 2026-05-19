"""
1.2 - Compute pairwise geodesic distances between all weather stations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FÓRMULAS UTILIZADAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. HAVERSINE → distance_km
   ─────────────────────────
   Calcula a distância do arco do grande círculo entre dois pontos na
   superfície de uma esfera, usando apenas latitude e longitude.

       a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
       distance_km = 2 · R · arcsin(√a)       R = 6 371 km

   Por quê: padrão para distâncias geodésicas em escala regional. O erro
   vs. métodos elipsoidais (Vincenty) é < 0.5% para distâncias de dezenas
   a centenas de km — aceitável para estações meteorológicas.

2. DELTA DE ALTITUDE → delta_altitude_m
   ──────────────────────────────────────
       delta_altitude_m = |altitude_from - altitude_to|

   Por quê: diferença bruta em metros preserva flexibilidade total para
   modelos downstream pesarem altitude conforme a variável-alvo
   (temperatura, precipitação, etc. têm sensibilidades diferentes).

3. DISTÂNCIA EFETIVA 3D → effective_distance_km
   ───────────────────────────────────────────────
   Estende o Haversine para três dimensões incorporando o desnível como
   penalidade horizontal equivalente, seguindo o padrão Meteonorm:

       effective_distance_km = √(distance_km² + (delta_altitude_m / 100)²)

   O fator /100 traduz altitude em distância: 100 m de desnível ≡ 1 km
   horizontal. Assim, 1 000 m de desnível adiciona 10 km à distância.

   Por quê: duas estações próximas horizontalmente mas em altitudes muito
   diferentes têm climas distintos (gradiente de temperatura ~6.5 °C/km,
   regimes de chuva orográficos, pressão). Essa métrica combina distância
   superficial e desnível em um único valor comparável, ideal para
   selecionar estações proxy em gap-filling e feature engineering de ML.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
    data/stations.parquet
        columns: code, name, altitude, latitude, longitude, state, start_operation

Output:
    data/station_distances.parquet
        columns: from_code (str), to_code (str),
                 distance_km (float32),
                 delta_altitude_m (float32),
                 effective_distance_km (float32)
        sorted by (from_code, effective_distance_km) — enables O(1) group lookup.

Usage after generation:
    distances = pd.read_parquet("data/station_distances.parquet")

    # Sort a subset of codes by effective distance from a reference station
    def sort_by_distance(ref_code: str, codes: list[str]) -> list[str]:
        group = distances.loc[ref_code]           # fast index lookup
        return group[group["to_code"].isin(codes)]["to_code"].tolist()
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

DATA_DIR = Path(__file__).parent / "data"
INPUT_PATH = DATA_DIR / "stations.parquet"
OUTPUT_PATH = DATA_DIR / "station_distances.parquet"

EARTH_RADIUS_KM = 6_371.0


def load_stations() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run 1.1_download_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(INPUT_PATH, columns=["code", "latitude", "longitude", "altitude"])
    df["latitude"]  = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["altitude"]  = df["altitude"].astype(float)
    df = df.dropna(subset=["latitude", "longitude", "altitude"]).drop_duplicates(subset="code")
    print(f"Stations loaded: {len(df)}")
    return df.reset_index(drop=True)


def compute_distances(stations: pd.DataFrame) -> pd.DataFrame:
    coords_rad = np.radians(stations[["latitude", "longitude"]].values)
    dist_matrix = haversine_distances(coords_rad) * EARTH_RADIUS_KM  # shape (N, N)

    altitudes = stations["altitude"].values
    delta_alt_matrix = np.abs(altitudes[:, None] - altitudes[None, :])  # shape (N, N)

    codes = stations["code"].values
    n = len(codes)

    # Build long-format — both directions so lookup is always on from_code
    from_codes = np.repeat(codes, n)
    to_codes   = np.tile(codes, n)
    distance_km      = dist_matrix.ravel()
    delta_altitude_m = delta_alt_matrix.ravel()

    effective_distance_km = np.sqrt(distance_km**2 + (delta_altitude_m / 100.0)**2)

    df = pd.DataFrame({
        "from_code":             from_codes,
        "to_code":               to_codes,
        "distance_km":           distance_km.astype(np.float32),
        "delta_altitude_m":      delta_altitude_m.astype(np.float32),
        "effective_distance_km": effective_distance_km.astype(np.float32),
    })

    # Remove self-distance rows
    df = df[df["from_code"] != df["to_code"]]

    # Sort so each from_code group is ordered nearest → farthest (by effective distance)
    df = df.sort_values(["from_code", "effective_distance_km"]).reset_index(drop=True)

    return df


def save(df: pd.DataFrame) -> None:
    # Set from_code as index — makes df.loc[code] instant
    df = df.set_index("from_code")
    df.to_parquet(OUTPUT_PATH, index=True, engine="pyarrow", compression="snappy")
    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    print(f"Saved: {OUTPUT_PATH}  ({len(df):,} rows, {size_mb:.2f} MB)")


def main():
    print("=== 1.2 Compute Station Distances ===\n")
    stations = load_stations()
    print("Computing pairwise distances...")
    df = compute_distances(stations)
    save(df)

    # Sanity check — show 5 nearest stations to the first code
    sample_code = stations["code"].iloc[0]
    result = df[df["from_code"] == sample_code].head(5)
    print(f"\nNearest 5 stations to code {sample_code} (by effective_distance_km):")
    print(result.to_string())


if __name__ == "__main__":
    main()
