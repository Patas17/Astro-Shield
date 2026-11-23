# 🛰️ AstroShield-CORE v1.0
**Sistema de Aprendizaje Automático para la Evaluación Predictiva de Riesgo de Disrupción e Intersección Orbital de NEOs sobre la Infraestructura Satelital Terrestre**

## 1. Requisitos del Entorno
* **Lenguaje:** Python 3.11+
* **Frameworks:** Scikit-Learn, DuckDB, Streamlit, Plotly, Pandas, Joblib

## 2. Instalación y Ejecución Local
```bash
# 1. Clonar el repositorio
git clone [https://github.com/TU_USUARIO/AstroShield-CORE.git](https://github.com/TU_USUARIO/AstroShield-CORE.git)
cd AstroShield-CORE

# 2. Crear y activar entorno virtual
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar pipeline ETL y entrenamiento (si se desea regenerar)
python src/data/preprocess.py
python src/models/train.py

# 5. Arrancar el Dashboard Interactivo
streamlit run app.py