import argparse
import os
import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLUMNAS_MODELO = [
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude",
    "shell_asset_density",
    "orbital_shell_LEO",
    "orbital_shell_MEO",
    "orbital_shell_GEO",
]


def train_model(data_path: str, output_dir: str):
    print(f"[INFO] Cargando dataset procesado: {data_path}...")
    df = pd.read_parquet(data_path)
    os.makedirs(output_dir, exist_ok=True)

    # Codificación One-Hot de concha orbital
    df_encoded = pd.get_dummies(df, columns=["orbital_shell"])
    for col in COLUMNAS_MODELO:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[COLUMNAS_MODELO]
    y = df_encoded["hazardous"].astype(int)

    # Partición 80/20 estratificada con semilla fija 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train):,} muestras | Test: {len(X_test):,} muestras")

    # Escalamiento estándar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento del modelo ganador (Random Forest Optimizado)
    print("[INFO] Entrenando RandomForestClassifier (n_estimators=150, max_depth=12, class_weight='balanced')...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)
    elapsed = time.time() - t0
    print(f"[INFO] Entrenamiento finalizado en {elapsed:.2f} segundos.")

    # Evaluación en Test Set
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n[RESULTADOS EN TEST SET]")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Macro F1-Score : {macro_f1:.4f} (Meta >= 0.85)")
    print(f"ROC-AUC Score  : {roc_auc:.4f} (Meta >= 0.90)")

    # Serialización de artefactos
    model_path = os.path.join(output_dir, "modelo_candidato.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[SUCCESS] Modelo guardado en {model_path}")
    print(f"[SUCCESS] Escalador guardado en {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/astroshield_master_processed.parquet")
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()
    train_model(args.data, args.output)