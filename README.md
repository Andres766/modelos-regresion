# Laboratorio de Minería de Datos — Regresión lineal múltiple

Tres modelos de regresión lineal múltiple desarrollados con la metodología **CRISP-DM**, exportados con `joblib` y servidos en una interfaz web hecha con **Streamlit**.

| Ejercicio | Se predice | A partir de | R² (prueba) | Error | Variable de mayor impacto |
|---|---|---|---|---|---|
| 1. Dólar | `Precio_Dolar` (COP) | Día, inflación, tasa de interés | 0,9963 | MSE = 2.376,97 | Día |
| 2. Glucosa | `Nivel_Glucosa` (mg/dL) | Edad, IMC, actividad física | 0,6814 | MSE = 233,69 | Edad |
| 3. Energía | `Consumo_Energia` (kWh) | Temperatura, hora, día de la semana | 0,8968 | RMSE = 20,72 kWh | Temperatura |

Métricas calculadas con el 20 % de los datos reservado para prueba (`random_state=42`). Los notebooks traen el análisis completo: limpieza, coeficientes, p-valores, VIF, validación cruzada, gráficas y conclusiones.

## Estructura

```
├── app.py                 # Interfaz web (Streamlit)
├── data/                  # Datasets del laboratorio (CSV)
├── notebooks/
│   ├── 01_dolar.ipynb     # Ejercicio 1
│   ├── 02_glucosa.ipynb   # Ejercicio 2
│   └── 03_energia.ipynb   # Ejercicio 3
├── modelos/               # Modelos exportados (.joblib), generados por los notebooks
├── figuras/               # Gráficas generadas por los notebooks (las usa la app)
├── informe/               # Informe escrito (INFORME.pdf, .docx y .md) y capturas de la app
├── scripts/               # Generación del informe (Markdown -> Word -> PDF, formato APA 7)
├── tests/test_app.py      # Pruebas de la app y de los modelos
├── requirements.txt       # Dependencias de la app (las usa Streamlit Cloud)
└── requirements-dev.txt   # + dependencias de los notebooks y las pruebas
```

## Ejecutar en local

Requiere Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements-dev.txt

streamlit run app.py              # abre http://localhost:8501
```

Para regenerar los modelos y las figuras, ejecuta los notebooks en orden (o desde la terminal):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
pytest -q
```

El informe se regenera con las cifras actuales de los modelos (el PDF requiere Microsoft Word en Windows):

```bash
python scripts/generar_informe.py
python scripts/exportar_informe.py
```

## Desplegar en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio de GitHub, **incluyendo** `data/`, `modelos/` y `figuras/`.
2. Entra a https://share.streamlit.io, inicia sesión con GitHub y pulsa **Create app**.
3. Elige el repositorio, la rama `main` y el archivo principal `app.py`.
4. En **Advanced settings** selecciona **Python 3.11**.
5. Pulsa **Deploy**. Streamlit instala `requirements.txt` y publica la app en un enlace `*.streamlit.app`.

> Los modelos se entrenaron con scikit-learn 1.9.1. `requirements.txt` fija esa misma versión para que la app los cargue sin problemas. Si se cambia la versión, hay que volver a ejecutar los notebooks.

## Decisiones importantes

- **No se eliminó ninguna fila:** no hay nulos ni duplicados, y los valores que marca la regla IQR son extremos pero posibles (por ejemplo, días muy calurosos o pacientes con hipoglucemia).
- **Hora en el ejercicio de energía:** se usa tal como viene (1 a 24) y no con seno y coseno. El consumo crece casi en línea recta a lo largo del día, y con validación cruzada la hora lineal obtiene R² = 0,900 frente a 0,785 de la versión cíclica.
- **Importancia de las variables:** se mide con el coeficiente estandarizado (β), que no depende de las unidades de cada variable.
- **Dólar:** la inflación (p = 0,096) y la tasa de interés (p = 0,797) no son estadísticamente significativas. Casi todo el precio lo explica la tendencia del día.
