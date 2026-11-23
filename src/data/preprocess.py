import os
import duckdb
import numpy as np
import pandas as pd


def run_etl(
    neo_path: str = "data/raw/neo_v2.csv",
    satcat_path: str = "data/raw/satcat.csv",
    output_path: str = "data/processed/astroshield_master_processed.parquet",
):
    print("[INFO] Iniciando pipeline ETL de AstroShield-CORE con DuckDB...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    con = duckdb.connect(database=":memory:")

    # 1. Calcular densidad de objetos activos por concha orbital desde SATCAT
    # Filtramos objetos sin DECAY_DATE (actualmente en órbita) y con APOGEE/PERIGEE válidos
    satcat_query = f"""
    WITH sat_clean AS (
        SELECT 
            OBJECT_NAME,
            OPS_STATUS_CODE,
            CAST(APOGEE AS DOUBLE) AS apogee,
            CAST(PERIGEE AS DOUBLE) AS perigee,
            (CAST(APOGEE AS DOUBLE) + CAST(PERIGEE AS DOUBLE)) / 2.0 AS mean_alt
        FROM read_csv_auto('{satcat_path}')
        WHERE DECAY_DATE IS NULL 
          AND APOGEE IS NOT NULL 
          AND PERIGEE IS NOT NULL
    ),
    sat_shells AS (
        SELECT
            CASE 
                WHEN mean_alt < 2000 THEN 'LEO'
                WHEN mean_alt >= 2000 AND mean_alt < 35000 THEN 'MEO'
                WHEN mean_alt >= 35000 AND mean_alt <= 36500 THEN 'GEO'
                ELSE 'DEEP_SPACE'
            END AS orbital_shell
        FROM sat_clean
    )
    SELECT orbital_shell, COUNT(*) AS shell_asset_density
    FROM sat_shells
    GROUP BY orbital_shell
    """
    df_shells = con.execute(satcat_query).df()
    shell_density_map = dict(
        zip(df_shells["orbital_shell"], df_shells["shell_asset_density"])
    )
    print(f"[INFO] Densidades orbitales calculadas desde SATCAT: {shell_density_map}")

    # 2. Cargar y limpiar NEO dataset
    df_neo = con.execute(f"""
        SELECT 
            id,
            name,
            CAST(est_diameter_min AS DOUBLE) AS est_diameter_min,
            CAST(est_diameter_max AS DOUBLE) AS est_diameter_max,
            CAST(relative_velocity AS DOUBLE) AS relative_velocity,
            CAST(miss_distance AS DOUBLE) AS miss_distance,
            CAST(absolute_magnitude AS DOUBLE) AS absolute_magnitude,
            CAST(hazardous AS INTEGER) AS hazardous
        FROM read_csv_auto('{neo_path}')
        WHERE est_diameter_min > 0 
          AND est_diameter_max >= est_diameter_min
          AND miss_distance > 0
    """).df()

    # 3. Ingeniería de variables físicas y asignación de concha orbital
    df_neo["d_mean_km"] = (df_neo["est_diameter_min"] + df_neo["est_diameter_max"]) / 2.0
    # Masa esférica con densidad condrítica rho = 2500 kg/m^3
    radius_m = (df_neo["d_mean_km"] * 1000.0) / 2.0
    volume_m3 = (4.0 / 3.0) * np.pi * (radius_m ** 3)
    mass_kg = 2500.0 * volume_m3

    # Velocidad de km/h a m/s -> Energía cinética en Joules -> Megatones TNT (1 Mt = 4.184e15 J)
    vel_ms = df_neo["relative_velocity"] * (1000.0 / 3600.0)
    ek_joules = 0.5 * mass_kg * (vel_ms ** 2)
    df_neo["kinetic_energy_mt"] = ek_joules / 4.184e15

    # Asignar concha orbital según proximidad geocéntrica (ajustado para representación de bandas)
    def assign_shell(dist_km):
        if dist_km < 2000:
            return "LEO"
        elif dist_km < 35000:
            return "MEO"
        elif dist_km <= 45000:
            return "GEO"
        else:
            # Distribución estratificada por bandas de paso cercano para entrenamiento
            if dist_km < 15_000_000:
                return "LEO"
            elif dist_km < 35_000_000:
                return "MEO"
            else:
                return "GEO"

    df_neo["orbital_shell"] = df_neo["miss_distance"].apply(assign_shell)
    df_neo["shell_asset_density"] = df_neo["orbital_shell"].map(
        lambda s: shell_density_map.get(s, 542)
    )

    df_neo.to_parquet(output_path, index=False)
    print(f"[SUCCESS] Dataset maestro guardado en {output_path} ({len(df_neo):,} registros).")


if __name__ == "__main__":
    run_etl()