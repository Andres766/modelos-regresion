"""Genera informe/INFORME.md y informe/portada.json con las cifras leídas de los modelos exportados y de los datos.

Uso (desde la raíz del proyecto, después de ejecutar los notebooks):
    python scripts/generar_informe.py
    python scripts/exportar_informe.py     # .md -> .docx + .pdf
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

RAIZ = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
INFORME = RAIZ / "informe"

PORTADA = {
    "titulo": "Regresión lineal múltiple para predecir el precio del dólar, la glucosa en sangre y el consumo de energía",
    "subtitulo": "Informe del laboratorio de minería de datos",
    "autores": ["Andrés Sebastián Cordoba"],
    "programa": "Programa de Ingeniería de Software",
    "universidad": "Universidad Cooperativa de Colombia",
    "asignatura": "Inteligencia de Negocios y Minería de Datos",
    "docente": "Ing. Cristian Camilo Ordoñez",
    "fecha": "Noviembre de 2026",
}


def n(valor, dec=2):
    """Formato colombiano con guion ASCII para los negativos."""
    texto = f"{valor:,.{dec}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def p_valor(p):
    return "< 0,001" if p < 0.001 else n(p, 3)


def atipicos(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)).sum())


def tabla(cabecera, filas):
    lineas = ["| " + " | ".join(cabecera) + " |", "|" + "---|" * len(cabecera)]
    lineas += ["| " + " | ".join(str(c) for c in f) + " |" for f in filas]
    return "\n".join(lineas)


def tabla_descriptiva(df, decimales):
    filas = []
    for c in df.columns:
        d = decimales.get(c, 2)
        filas.append([f"`{c}`", n(df[c].mean(), d), n(df[c].std(), d), n(df[c].min(), d), n(df[c].max(), d),
                      atipicos(df[c])])
    return tabla(["Variable", "Media", "Desv. estándar", "Mínimo", "Máximo", "Atípicos (IQR)"], filas)


def tabla_coeficientes(a):
    beta = a["coef_estandarizados"]
    total = sum(abs(b) for b in beta.values())
    orden = sorted(a["variables"], key=lambda v: abs(beta[v]), reverse=True)
    filas = [["Intercepto (β₀)", n(a["intercepto"], 4), "—", "—", "—"]]
    filas += [[f"`{v}`", n(a["coeficientes"][v], 4), n(beta[v], 3), f"{n(100 * abs(beta[v]) / total, 1)} %",
               p_valor(a["p_valores"][v])] for v in orden]
    return tabla(["Término", "Coeficiente", "β estandarizado", "Importancia", "p-valor"], filas), orden


def tabla_metricas(a, unidad):
    tr, te = a["metricas_entrenamiento"], a["metricas_prueba"]
    filas = [
        [f"MSE ({unidad})²", n(tr["MSE"]), n(te["MSE"])],
        [f"RMSE ({unidad})", n(tr["RMSE"]), n(te["RMSE"])],
        [f"MAE ({unidad})", n(tr["MAE"]), n(te["MAE"])],
        ["R²", n(tr["R2"], 4), n(te["R2"], 4)],
    ]
    return tabla(["Métrica", "Entrenamiento", "Prueba"], filas)


A = {k: joblib.load(RAIZ / "modelos" / f"modelo_{k}.joblib") for k in ("dolar", "glucosa", "energia")}
D = {
    "dolar": pd.read_csv(RAIZ / "data" / "dolar_data.csv"),
    "glucosa": pd.read_csv(RAIZ / "data" / "glucosa_data.csv"),
    "energia": pd.read_csv(RAIZ / "data" / "energia_data.csv"),
}
dol, glu, ene = A["dolar"], A["glucosa"], A["energia"]
mdol, mglu, mene = dol["metricas_prueba"], glu["metricas_prueba"], ene["metricas_prueba"]

# Correlaciones y efectos que se citan en el texto
r = {k: D[k].corr()[A[k]["objetivo"]] for k in D}
q = lambda k, v: D[k][v].quantile(0.75) - D[k][v].quantile(0.25)
efecto_rango = lambda k, v: A[k]["coeficientes"][v] * q(k, v)

# Comparación de la codificación de la hora (misma validación cruzada que el notebook)
df_e = D["energia"]
hora_rad = 2 * np.pi * df_e["Hora"] / 24
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_lineal = cross_val_score(LinearRegression(), df_e[ene["variables"]], df_e["Consumo_Energia"], cv=kf, scoring="r2").mean()
r2_ciclica = cross_val_score(
    LinearRegression(),
    df_e[["Temperatura", "Dia_Semana"]].assign(s=np.sin(hora_rad), c=np.cos(hora_rad)),
    df_e["Consumo_Energia"], cv=kf, scoring="r2",
).mean()
por_hora = df_e.groupby("Hora")["Consumo_Energia"].mean()
por_dia = df_e.groupby("Dia_Semana")["Consumo_Energia"].mean()

tc_dol, orden_dol = tabla_coeficientes(dol)
tc_glu, orden_glu = tabla_coeficientes(glu)
tc_ene, orden_ene = tabla_coeficientes(ene)
cv = lambda a: f"{n(a['r2_validacion_cruzada']['media'], 4)} ± {n(a['r2_validacion_cruzada']['desviacion'], 4)}"
ec = lambda a: f"{a['objetivo']} = {n(a['intercepto'], 4)} " + " ".join(
    f"{'+' if c >= 0 else '−'} {n(abs(c), 4)} × {v}" for v, c in a["coeficientes"].items())

md = f"""# {PORTADA['titulo']}

{PORTADA['subtitulo']}

## Resumen

Este informe presenta un laboratorio de minería de datos desarrollado con la metodología CRISP-DM. Se ajustaron tres modelos de regresión lineal múltiple para predecir variables cuantitativas en contextos distintos: el precio del dólar a partir del día, la inflación y la tasa de interés; el nivel de glucosa en sangre a partir de la edad, el índice de masa corporal y la actividad física; y el consumo de energía eléctrica a partir de la temperatura, la hora y el día de la semana. Cada modelo se entrenó con el 80 % de los datos y se evaluó con el 20 % restante. En los datos de prueba, el modelo del dólar obtuvo un R² de {n(mdol['R2'], 4)} (MSE = {n(mdol['MSE'])}), el de glucosa un R² de {n(mglu['R2'], 4)} (MSE = {n(mglu['MSE'])}) y el de energía un R² de {n(mene['R2'], 4)} (RMSE = {n(mene['RMSE'])} kWh). Las variables de mayor impacto fueron el día, la edad y la temperatura, respectivamente. Los modelos se exportaron con joblib y se integraron en una interfaz web desarrollada con Streamlit, donde el usuario elige el ejercicio, escribe los valores de las variables y obtiene la predicción.

*Palabras clave:* minería de datos, CRISP-DM, regresión lineal múltiple, scikit-learn, Streamlit

## Introducción

La regresión lineal múltiple es uno de los modelos más usados para predecir una variable cuantitativa a partir de varias variables explicativas. Su principal ventaja es la interpretabilidad: cada coeficiente indica cuánto cambia la predicción cuando una variable aumenta en una unidad y las demás se mantienen constantes (James et al., 2021).

El objetivo del laboratorio es aplicar las fases de la metodología CRISP-DM (Chapman et al., 2000) para integrar un modelo de regresión lineal en tres contextos, interpretar sus resultados, evaluar su desempeño, exportarlo y construir una interfaz web básica que permita hacer predicciones. Los tres ejercicios son:

1. Predicción del precio del dólar (`dolar_data.csv`).
2. Predicción del nivel de glucosa en sangre (`glucosa_data.csv`).
3. Predicción del consumo de energía eléctrica (`energia_data.csv`).

## Metodología

El trabajo siguió las seis fases de CRISP-DM. Las cinco primeras se aplicaron a cada ejercicio en un notebook de Jupyter independiente, y la fase de despliegue corresponde a la exportación de los modelos y a la interfaz web.

### Comprensión del negocio y de los datos

En cada ejercicio se definió la variable a predecir y las preguntas a responder. Luego se revisaron los tipos de dato, los valores nulos, los registros duplicados, las estadísticas descriptivas y la correlación de Pearson entre las variables. La Tabla 1 resume los tres conjuntos de datos.

*Tabla 1. Conjuntos de datos del laboratorio*

{tabla(["Ejercicio", "Archivo", "Filas", "Variables independientes", "Variable dependiente"], [
    ["Dólar", "dolar_data.csv", n(len(D['dolar']), 0), "Dia, Inflacion, Tasa_interes", "Precio_Dolar (COP)"],
    ["Glucosa", "glucosa_data.csv", n(len(D['glucosa']), 0), "Edad, IMC, Actividad_Fisica", "Nivel_Glucosa (mg/dL)"],
    ["Energía", "energia_data.csv", n(len(D['energia']), 0), "Temperatura, Hora, Dia_Semana", "Consumo_Energia (kWh)"],
])}

*Nota.* Ninguno de los tres conjuntos tiene valores nulos ni filas duplicadas.

### Preparación de los datos

Los valores atípicos se identificaron con la regla del rango intercuartílico (IQR): un valor es atípico si está por debajo de Q1 − 1,5·IQR o por encima de Q3 + 1,5·IQR. Antes de eliminar un registro se verificó si el valor era físicamente imposible (un error de captura) o un caso extremo pero real. En los tres conjuntos los valores marcados resultaron posibles, por lo que se conservaron todas las filas. Luego, los datos se dividieron en un 80 % para entrenamiento y un 20 % para prueba, con una semilla fija (`random_state = 42`) para que los resultados sean reproducibles.

### Modelado

Se ajustó un modelo de regresión lineal múltiple por mínimos cuadrados ordinarios con la clase `LinearRegression` de scikit-learn (Pedregosa et al., 2011), de la forma ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃. El mismo modelo se ajustó con statsmodels (Seabold & Perktold, 2010) para obtener los p-valores de cada coeficiente y el factor de inflación de la varianza (VIF), que mide la multicolinealidad entre las variables.

### Evaluación

El desempeño se midió en los datos de prueba con el error cuadrático medio (MSE), su raíz (RMSE), el error absoluto medio (MAE) y el coeficiente de determinación (R²). Para descartar sobreajuste se compararon las métricas de entrenamiento y de prueba, y se calculó el R² con validación cruzada de cinco pliegues. La importancia de cada variable se midió con el **coeficiente estandarizado** (β), que expresa el efecto en desviaciones estándar y por eso permite comparar variables con unidades distintas.

## Ejercicio 1: predicción del precio del dólar

### Comprensión y preparación de los datos

El conjunto tiene {n(len(D['dolar']), 0)} días consecutivos. La Tabla 2 muestra sus estadísticas descriptivas. La regla IQR marca {atipicos(D['dolar']['Inflacion'])} valores de inflación y {atipicos(D['dolar']['Tasa_interes'])} de tasa de interés, pero todos están en rangos económicamente posibles (inflación entre {n(D['dolar']['Inflacion'].min() * 100, 2)} % y {n(D['dolar']['Inflacion'].max() * 100, 2)} %), así que se conservaron las {n(len(D['dolar']), 0)} filas. La correlación del precio con el día es de r = {n(r['dolar']['Dia'], 3)}, mientras que con la inflación es de {n(r['dolar']['Inflacion'], 3)} y con la tasa de interés de {n(r['dolar']['Tasa_interes'], 3)}.

*Tabla 2. Estadísticas descriptivas del conjunto del dólar*

{tabla_descriptiva(D['dolar'], {'Dia': 0, 'Inflacion': 4})}

### Modelo e interpretación de los coeficientes

La ecuación ajustada es:

`{ec(dol)}`

*Tabla 3. Coeficientes del modelo del dólar*

{tc_dol}

*Nota.* El β estandarizado es el coeficiente multiplicado por la desviación estándar de la variable y dividido por la del precio. La importancia es el porcentaje de |β| sobre la suma de los tres.

- **Día:** cada día adicional, el precio sube {n(dol['coeficientes']['Dia'])} COP, manteniendo constantes las demás variables. Es significativo (p {p_valor(dol['p_valores']['Dia'])}) y concentra el {n(100 * abs(dol['coef_estandarizados']['Dia']) / sum(abs(b) for b in dol['coef_estandarizados'].values()), 1)} % del impacto.
- **Inflación:** por cada punto porcentual adicional (0,01), el precio baja {n(abs(dol['coeficientes']['Inflacion']) * 0.01)} COP. El efecto **no es estadísticamente significativo** (p = {p_valor(dol['p_valores']['Inflacion'])}).
- **Tasa de interés:** por cada punto adicional, el precio baja {n(abs(dol['coeficientes']['Tasa_interes']))} COP. Tampoco es significativa (p = {p_valor(dol['p_valores']['Tasa_interes'])}).

Los VIF de las tres variables son cercanos a 1, así que no hay multicolinealidad y los coeficientes se pueden interpretar por separado.

### Desempeño del modelo

*Tabla 4. Métricas del modelo del dólar*

{tabla_metricas(dol, 'COP')}

El modelo explica el {n(mdol['R2'] * 100, 1)} % de la variación del precio en los datos de prueba. El RMSE de {n(mdol['RMSE'])} COP equivale al {n(mdol['RMSE'] / D['dolar']['Precio_Dolar'].mean() * 100, 1)} % del precio promedio ({n(D['dolar']['Precio_Dolar'].mean())} COP). El R² con validación cruzada fue de {cv(dol)}, muy parecido al de prueba, por lo que no hay sobreajuste.

### Visualizaciones

La Figura 1 muestra la relación de cada variable independiente con el precio. El día forma una línea casi perfecta, mientras que la inflación y la tasa de interés aparecen como nubes horizontales sin tendencia. La Figura 2 compara los valores reales con los predichos.

![Figura 1. Relación de cada variable independiente con el precio del dólar](../figuras/dolar_dispersion.png)

![Figura 2. Valores reales frente a predichos y residuos del modelo del dólar](../figuras/dolar_evaluacion.png)

## Ejercicio 2: predicción del nivel de glucosa en sangre

### Comprensión y preparación de los datos

El conjunto tiene {n(len(D['glucosa']), 0)} pacientes (Tabla 5). La regla IQR marca {atipicos(D['glucosa']['IMC'])} valores de IMC y {atipicos(D['glucosa']['Nivel_Glucosa'])} de glucosa. Son valores extremos pero clínicamente posibles (delgadez severa, obesidad, hipoglucemia o hiperglucemia), y eliminarlos haría que el modelo ignorara justo a los pacientes de mayor riesgo. Por eso se conservaron todas las filas.

*Tabla 5. Estadísticas descriptivas del conjunto de glucosa*

{tabla_descriptiva(D['glucosa'], {'Edad': 1, 'Actividad_Fisica': 1})}

### Modelo e interpretación de los coeficientes

La ecuación ajustada es:

`{ec(glu)}`

*Tabla 6. Coeficientes del modelo de glucosa*

{tc_glu}

Las tres variables son estadísticamente significativas (p < 0,05):

- **Edad:** cada año adicional sube la glucosa {n(glu['coeficientes']['Edad'])} mg/dL.
- **IMC:** cada punto adicional sube la glucosa {n(glu['coeficientes']['IMC'])} mg/dL.
- **Actividad física:** cada hora semanal de ejercicio baja la glucosa {n(abs(glu['coeficientes']['Actividad_Fisica']))} mg/dL.

### Variable de mayor impacto

La **edad** es la variable de mayor impacto: tiene el coeficiente estandarizado más alto (β = {n(glu['coef_estandarizados']['Edad'], 3)}), concentra el {n(100 * abs(glu['coef_estandarizados']['Edad']) / sum(abs(b) for b in glu['coef_estandarizados'].values()), 1)} % de la importancia y es la más correlacionada con la glucosa (r = {n(r['glucosa']['Edad'], 2)}). Para comparar en términos prácticos: pasar del percentil 25 al percentil 75 de cada variable cambia la glucosa en {n(efecto_rango('glucosa', 'Edad'), 1)} mg/dL en el caso de la edad, en {n(efecto_rango('glucosa', 'Actividad_Fisica'), 1)} mg/dL en el de la actividad física y en {n(efecto_rango('glucosa', 'IMC'), 1)} mg/dL en el del IMC (Figura 4). Aunque el coeficiente del IMC por unidad es parecido al de la edad, el IMC varía mucho menos entre pacientes, y por eso su impacto total es menor.

### Desempeño del modelo

*Tabla 7. Métricas del modelo de glucosa*

{tabla_metricas(glu, 'mg/dL')}

El modelo explica el {n(mglu['R2'] * 100, 1)} % de la variación de la glucosa, con un error típico (RMSE) de {n(mglu['RMSE'])} mg/dL, equivalente al {n(mglu['RMSE'] / D['glucosa']['Nivel_Glucosa'].mean() * 100, 1)} % del promedio. El R² con validación cruzada fue de {cv(glu)}, sin señales de sobreajuste. El {n((1 - mglu['R2']) * 100, 0)} % restante de la variación depende de factores que no están en los datos, como la dieta, la genética o la medicación.

### Visualizaciones

![Figura 3. Relación de cada variable independiente con el nivel de glucosa](../figuras/glucosa_dispersion.png)

![Figura 4. Coeficientes estandarizados del modelo de glucosa](../figuras/glucosa_importancia.png)

![Figura 5. Valores reales frente a predichos y residuos del modelo de glucosa](../figuras/glucosa_evaluacion.png)

## Ejercicio 3: predicción del consumo de energía eléctrica

### Comprensión y preparación de los datos

El conjunto tiene {n(len(D['energia']), 0)} registros (Tabla 8). La regla IQR marca {atipicos(D['energia']['Temperatura'])} temperaturas y {atipicos(D['energia']['Consumo_Energia'])} consumos. Corresponden a días muy fríos o muy calurosos con consumos coherentes con esa temperatura, así que se conservaron todas las filas.

*Tabla 8. Estadísticas descriptivas del conjunto de energía*

{tabla_descriptiva(D['energia'], {'Hora': 1, 'Dia_Semana': 1})}

### Tratamiento de la hora

El enunciado indica que la hora es una variable "ciclada" (después de la 24 vuelve la 1). Una práctica común con este tipo de variables es codificarlas con seno y coseno, para que el modelo entienda que la hora 24 y la hora 1 están cerca. Antes de decidir, se analizó el consumo promedio por hora (Figura 6): sube casi en línea recta, de {n(por_hora.iloc[0], 1)} kWh en la hora 1 a {n(por_hora.iloc[-1], 1)} kWh en la hora 24, sin volver a bajar. Por eso la codificación cíclica no se ajusta a estos datos, lo que se confirmó con validación cruzada (Tabla 9).

*Tabla 9. Comparación de dos formas de usar la hora*

{tabla(["Codificación de la hora", "R² medio (validación cruzada, 5 pliegues)"], [
    ["Hora lineal (1 a 24)", n(r2_lineal, 4)],
    ["Hora cíclica (seno y coseno)", n(r2_ciclica, 4)],
])}

Se usó la hora tal como viene (1 a 24). El día de la semana también muestra una tendencia casi lineal (el consumo promedio baja de {n(por_dia.iloc[0], 1)} kWh el lunes a {n(por_dia.iloc[-1], 1)} kWh el domingo), así que se dejó como variable numérica.

![Figura 6. Consumo medio de energía por hora y por día de la semana](../figuras/energia_promedios.png)

### Modelo e interpretación de los coeficientes

La ecuación ajustada es:

`{ec(ene)}`

*Tabla 10. Coeficientes del modelo de energía*

{tc_ene}

Las tres variables son estadísticamente significativas (p < 0,05):

- **Temperatura:** cada grado Celsius adicional sube el consumo {n(ene['coeficientes']['Temperatura'])} kWh.
- **Hora:** cada hora más tarde en el día sube el consumo {n(ene['coeficientes']['Hora'])} kWh.
- **Día de la semana:** cada día más adelante en la semana baja el consumo {n(abs(ene['coeficientes']['Dia_Semana']))} kWh, así que el domingo se consumen unos {n(abs(ene['coeficientes']['Dia_Semana']) * 6, 1)} kWh menos que el lunes.

### Variable de mayor impacto

La **temperatura** es la variable de mayor impacto (β = {n(ene['coef_estandarizados']['Temperatura'], 3)}, {n(100 * abs(ene['coef_estandarizados']['Temperatura']) / sum(abs(b) for b in ene['coef_estandarizados'].values()), 1)} % de la importancia), seguida de la hora (β = {n(ene['coef_estandarizados']['Hora'], 3)}). El día de la semana tiene un efecto pequeño (β = {n(ene['coef_estandarizados']['Dia_Semana'], 3)}). Las Figuras 7 y 8 muestran las relaciones más importantes mediante gráficas de dispersión y de correlación, y la Figura 9 compara el impacto de las variables.

### Desempeño del modelo

*Tabla 11. Métricas del modelo de energía*

{tabla_metricas(ene, 'kWh')}

El modelo explica el {n(mene['R2'] * 100, 1)} % de la variación del consumo, con un **RMSE de {n(mene['RMSE'])} kWh**, equivalente al {n(mene['RMSE'] / D['energia']['Consumo_Energia'].mean() * 100, 1)} % del consumo promedio ({n(D['energia']['Consumo_Energia'].mean())} kWh). El R² con validación cruzada fue de {cv(ene)}, sin señales de sobreajuste.

### Visualizaciones

![Figura 7. Relación de cada variable independiente con el consumo de energía](../figuras/energia_dispersion.png)

![Figura 8. Matriz de correlación del conjunto de energía](../figuras/energia_correlacion.png)

![Figura 9. Coeficientes estandarizados del modelo de energía](../figuras/energia_importancia.png)

## Exportación de los modelos

Un modelo entrenado se puede guardar en un archivo para usarlo después sin volver a entrenarlo. En Python hay dos opciones habituales (Tabla 12).

*Tabla 12. Comparación de pickle y joblib*

{tabla(["Aspecto", "pickle", "joblib"], [
    ["Qué es", "Módulo estándar de Python", "Librería usada por scikit-learn"],
    ["Guardar", "`pickle.dump(modelo, archivo)`", "`joblib.dump(modelo, ruta)`"],
    ["Cargar", "`pickle.load(archivo)`", "`joblib.load(ruta)`"],
    ["Ventaja", "No requiere instalar nada", "Más eficiente con arreglos de NumPy"],
])}

Se eligió **joblib**, siguiendo la recomendación de la documentación de scikit-learn (scikit-learn developers, 2026). Cada archivo (`modelo_dolar.joblib`, `modelo_glucosa.joblib` y `modelo_energia.joblib`) guarda el modelo entrenado junto con los nombres de las variables, las métricas, los coeficientes, los p-valores y el rango de los datos de entrenamiento, que la interfaz web usa para mostrar la información del modelo y advertir cuando un valor está fuera de ese rango. Al cargar cada archivo se comprobó que el modelo recuperado produce exactamente las mismas predicciones que el original. Como un modelo guardado solo es compatible con la versión de scikit-learn con la que se creó, esa versión ({dol['version_sklearn']}) queda registrada en el archivo y fijada en `requirements.txt`.

## Interfaz web

La interfaz se desarrolló con Streamlit (Snowflake Inc., 2026), una de las tecnologías sugeridas en el enunciado. Al iniciar, carga los tres modelos exportados. El usuario:

1. Elige el ejercicio (Dólar, Glucosa o Energía) en el panel lateral.
2. Escribe con el teclado los valores de las variables independientes.
3. Pulsa **Calcular predicción** y ve el valor estimado, su margen de error típico (± RMSE) y el cálculo paso a paso con la ecuación del modelo.

Además, la interfaz tiene pestañas con el desempeño del modelo (R², MSE, RMSE y MAE), la interpretación de los coeficientes, las gráficas y los datos. La Figura 10 muestra una predicción para un paciente de 60 años con IMC de 31,5 y 2 horas semanales de ejercicio, y la Figura 11 la interpretación de los coeficientes. La interfaz se verificó con pruebas automáticas que comprueban que la predicción de cada ejercicio coincide con la ecuación del modelo.

![Figura 10. Predicción del nivel de glucosa en la interfaz web](capturas/app_glucosa.png)

![Figura 11. Pestaña de coeficientes e interpretación de la interfaz web](capturas/app_coeficientes.png)

## Conclusiones

*Tabla 13. Resumen de los tres modelos (datos de prueba)*

{tabla(["Ejercicio", "R²", "MSE", "RMSE", "Variable de mayor impacto"], [
    ["Dólar", n(mdol['R2'], 4), n(mdol['MSE']), f"{n(mdol['RMSE'])} COP", "Día"],
    ["Glucosa", n(mglu['R2'], 4), n(mglu['MSE']), f"{n(mglu['RMSE'])} mg/dL", "Edad"],
    ["Energía", n(mene['R2'], 4), n(mene['MSE']), f"{n(mene['RMSE'])} kWh", "Temperatura"],
])}

1. Los tres modelos de regresión lineal múltiple generalizan bien: las métricas de entrenamiento, de prueba y de validación cruzada son casi iguales, así que no hay sobreajuste.
2. El modelo del dólar tiene el mejor ajuste (R² = {n(mdol['R2'], 4)}), pero se debe casi por completo a la tendencia del día. La inflación y la tasa de interés no son significativas en estos datos, por lo que el modelo funciona en la práctica como una tendencia en el tiempo y no se debe usar para extrapolar a fechas lejanas.
3. En la glucosa, la edad es la variable de mayor impacto, seguida de la actividad física y del IMC. El modelo explica el {n(mglu['R2'] * 100, 0)} % de la variación: es útil para estimar tendencias, pero una parte importante depende de factores que no están en los datos.
4. En el consumo de energía, la temperatura es la variable de mayor impacto, seguida de la hora. Analizar los datos antes de modelar fue decisivo: la codificación cíclica de la hora, que suele recomendarse para este tipo de variables, habría bajado el R² de {n(r2_lineal, 2)} a {n(r2_ciclica, 2)}.
5. Los coeficientes estandarizados permitieron comparar el impacto de variables con unidades distintas, y los p-valores permitieron distinguir los efectos reales de los que podrían deberse al azar.
6. Exportar los modelos con joblib e integrarlos en una interfaz web completa el ciclo de CRISP-DM: los modelos dejan de estar solo en un notebook y cualquier usuario puede usarlos para hacer predicciones.

## Referencias

- Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An introduction to statistical learning: With applications in R* (2.ª ed.). Springer. https://doi.org/10.1007/978-1-0716-1418-1
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- scikit-learn developers. (2026). *Model persistence*. https://scikit-learn.org/stable/model_persistence.html
- Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. En *Proceedings of the 9th Python in Science Conference* (pp. 92–96). https://doi.org/10.25080/Majora-92bf1922-011
- Snowflake Inc. (2026). *Streamlit documentation*. https://docs.streamlit.io
"""

INFORME.mkdir(exist_ok=True)
(INFORME / "INFORME.md").write_text(md, encoding="utf-8")
(INFORME / "portada.json").write_text(json.dumps(PORTADA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("INFORME.md:", len(md.split()), "palabras")
print(f"hora lineal {r2_lineal:.4f} ciclica {r2_ciclica:.4f}")
