"""Interfaz web del laboratorio: predicciones con los tres modelos de regresión lineal múltiple."""

from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

RAIZ = Path(__file__).parent
DIAS = {1: "lunes", 2: "martes", 3: "miércoles", 4: "jueves", 5: "viernes", 6: "sábado", 7: "domingo"}

# Configuración de cada ejercicio. "limites" acota lo que se puede escribir;
# "paso" es el cambio realista que se usa para explicar cada coeficiente.
EJERCICIOS = {
    "Dólar": {
        "prefijo": "dolar",
        "archivo": "dolar_data.csv",
        "titulo": "Predicción del precio del dólar",
        "pregunta": "¿Cuánto costará el dólar según el día, la inflación y la tasa de interés?",
        "resultado": "Precio estimado del dólar",
        "decimales": 2,
        "campos": {
            "Dia": {"etiqueta": "Día (número de día de la serie)", "limites": (1, 10_000), "entero": True,
                    "paso": 1, "texto_paso": "1 día más"},
            "Inflacion": {"etiqueta": "Inflación diaria (fracción: 0,02 = 2 %)", "limites": (0.0, 1.0),
                          "formato": "%.4f", "incremento": 0.001,
                          "paso": 0.01, "texto_paso": "1 punto porcentual más de inflación"},
            "Tasa_interes": {"etiqueta": "Tasa de interés diaria (%)", "limites": (0.0, 100.0),
                             "formato": "%.2f", "incremento": 0.1,
                             "paso": 1, "texto_paso": "1 punto más de tasa de interés"},
        },
    },
    "Glucosa": {
        "prefijo": "glucosa",
        "archivo": "glucosa_data.csv",
        "titulo": "Predicción del nivel de glucosa en sangre",
        "pregunta": "¿Qué nivel de glucosa se espera según la edad, el IMC y la actividad física?",
        "resultado": "Nivel de glucosa estimado",
        "decimales": 1,
        "campos": {
            "Edad": {"etiqueta": "Edad (años)", "limites": (1, 120), "entero": True,
                     "paso": 1, "texto_paso": "1 año más de edad"},
            "IMC": {"etiqueta": "Índice de masa corporal (IMC)", "limites": (10.0, 70.0),
                    "formato": "%.1f", "incremento": 0.1,
                    "paso": 1, "texto_paso": "1 punto más de IMC"},
            "Actividad_Fisica": {"etiqueta": "Actividad física (horas por semana)", "limites": (0, 80), "entero": True,
                                 "paso": 1, "texto_paso": "1 hora más de ejercicio a la semana"},
        },
    },
    "Energía": {
        "prefijo": "energia",
        "archivo": "energia_data.csv",
        "titulo": "Predicción del consumo de energía eléctrica",
        "pregunta": "¿Cuánta energía se consumirá según la temperatura, la hora y el día de la semana?",
        "resultado": "Consumo de energía estimado",
        "decimales": 1,
        "campos": {
            "Temperatura": {"etiqueta": "Temperatura (°C)", "limites": (-30.0, 60.0),
                            "formato": "%.1f", "incremento": 0.5,
                            "paso": 1, "texto_paso": "1 °C más de temperatura"},
            "Hora": {"etiqueta": "Hora del día (1 a 24)", "limites": (1, 24), "entero": True,
                     "paso": 1, "texto_paso": "1 hora más tarde en el día"},
            "Dia_Semana": {"etiqueta": "Día de la semana (1 = lunes … 7 = domingo)", "limites": (1, 7), "entero": True,
                           "paso": 1, "texto_paso": "1 día más adelante en la semana"},
        },
    },
}


def numero(valor, decimales=2):
    """Formato colombiano: punto para miles y coma para decimales."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


@st.cache_resource
def cargar_modelo(prefijo):
    return joblib.load(RAIZ / "modelos" / f"modelo_{prefijo}.joblib")


@st.cache_data
def cargar_datos(archivo):
    return pd.read_csv(RAIZ / "data" / archivo)


def mostrar_figura(nombre, pie, ancho="stretch"):
    ruta = RAIZ / "figuras" / nombre
    if ruta.exists():
        st.image(str(ruta), caption=pie, width=ancho)


st.set_page_config(page_title="Regresión lineal · Laboratorio", page_icon="📈", layout="wide")

with st.sidebar:
    st.title("📈 Regresión lineal múltiple")
    st.caption("Laboratorio de Minería de Datos · CRISP-DM")
    nombre = st.radio("Elige el ejercicio", list(EJERCICIOS), key="ejercicio")
    st.divider()
    st.markdown(
        "**Cómo usarla**\n\n"
        "1. Elige el ejercicio.\n"
        "2. Escribe los valores de las variables.\n"
        "3. Pulsa **Calcular predicción**.\n\n"
        "Debajo puedes ver el desempeño del modelo, la interpretación de los coeficientes y las gráficas."
    )

ej = EJERCICIOS[nombre]
art = cargar_modelo(ej["prefijo"])
datos = cargar_datos(ej["archivo"])
unidad = art["unidad"]
dec = ej["decimales"]

if art["version_sklearn"] != sklearn.__version__:
    st.warning(
        f"El modelo se entrenó con scikit-learn {art['version_sklearn']} y aquí está instalada la "
        f"{sklearn.__version__}. Instala la misma versión indicada en requirements.txt."
    )

st.title(ej["titulo"])
st.write(ej["pregunta"])

col_form, col_resultado = st.columns([1, 1.15], gap="large")

with col_form:
    with st.form(f"form_{ej['prefijo']}"):
        st.subheader("Datos de entrada")
        entrada = {}
        for variable, campo in ej["campos"].items():
            minimo, maximo = campo["limites"]
            mediana = art["rangos"][variable]["mediana"]
            if campo.get("entero"):
                entrada[variable] = st.number_input(
                    campo["etiqueta"], min_value=int(minimo), max_value=int(maximo),
                    value=int(round(mediana)), step=1, key=f"{ej['prefijo']}_{variable}",
                )
            else:
                entrada[variable] = st.number_input(
                    campo["etiqueta"], min_value=float(minimo), max_value=float(maximo),
                    value=float(round(mediana, 4)), step=campo["incremento"], format=campo["formato"],
                    key=f"{ej['prefijo']}_{variable}",
                )
            rango = art["rangos"][variable]
            decimales_rango = 4 if variable == "Inflacion" else (0 if campo.get("entero") else 2)
            st.caption(f"Rango en los datos: {numero(rango['min'], decimales_rango)} a {numero(rango['max'], decimales_rango)}")
        enviado = st.form_submit_button("Calcular predicción", type="primary", width="stretch")

if enviado:
    fila = pd.DataFrame([entrada], columns=art["variables"])
    st.session_state[f"resultado_{ej['prefijo']}"] = {
        "entrada": entrada,
        "prediccion": float(art["modelo"].predict(fila)[0]),
    }

with col_resultado:
    resultado = st.session_state.get(f"resultado_{ej['prefijo']}")
    if resultado is None:
        st.info("Escribe los valores y pulsa **Calcular predicción** para ver el resultado.")
    else:
        pred = resultado["prediccion"]
        rmse = art["metricas_prueba"]["RMSE"]
        with st.container(border=True):
            st.metric(ej["resultado"], f"{numero(pred, dec)} {unidad}")
            st.caption(
                f"Error típico del modelo (RMSE): ± {numero(rmse, dec)} {unidad} → "
                f"valor probable entre {numero(pred - rmse, dec)} y {numero(pred + rmse, dec)} {unidad}."
            )

        fuera = [
            v for v, x in resultado["entrada"].items()
            if not art["rangos"][v]["min"] <= x <= art["rangos"][v]["max"]
        ]
        if fuera:
            st.warning(
                "Estos valores están fuera del rango con el que se entrenó el modelo: "
                + ", ".join(fuera) + ". La predicción es una extrapolación y puede ser menos confiable."
            )
        if pred < 0:
            st.warning("El modelo devuelve un valor negativo, que no tiene sentido físico: revisa los datos de entrada.")

        st.markdown("**Cómo se calculó** (ecuación del modelo con tus datos)")
        filas = [{"Término": "Intercepto (β₀)", "Valor ingresado": "—", "Coeficiente": numero(art["intercepto"], 4),
                  "Aporte": numero(art["intercepto"], dec)}]
        for variable in art["variables"]:
            x = resultado["entrada"][variable]
            coef = art["coeficientes"][variable]
            valor = numero(x, 4 if variable == "Inflacion" else (0 if isinstance(x, int) else 2))
            if variable == "Dia_Semana":
                valor += f" ({DIAS[x]})"
            filas.append({
                "Término": variable,
                "Valor ingresado": valor,
                "Coeficiente": numero(coef, 4),
                "Aporte": numero(coef * x, dec),
            })
        filas.append({"Término": "Predicción", "Valor ingresado": "", "Coeficiente": "", "Aporte": numero(pred, dec)})
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

st.divider()
tab_desempeno, tab_coef, tab_graficas, tab_datos = st.tabs(
    ["📊 Desempeño", "🧮 Coeficientes e interpretación", "📈 Gráficas", "🗂️ Datos"]
)

with tab_desempeno:
    m = art["metricas_prueba"]
    cv = art["r2_validacion_cruzada"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", numero(m["R2"], 4), help="Proporción de la variación explicada por el modelo (1 = perfecto).")
    c2.metric("MSE", numero(m["MSE"], 2), help=f"Error cuadrático medio, en ({unidad})².")
    c3.metric("RMSE", f"{numero(m['RMSE'], 2)} {unidad}", help="Raíz del MSE: error típico en las unidades originales.")
    c4.metric("MAE", f"{numero(m['MAE'], 2)} {unidad}", help="Error absoluto medio.")
    st.caption(
        f"Métricas calculadas con el 20 % de los datos que el modelo no vio al entrenar "
        f"({numero(art['filas']['prueba'], 0)} filas; entrenamiento: {numero(art['filas']['entrenamiento'], 0)}). "
        f"R² con validación cruzada de 5 pliegues: {numero(cv['media'], 4)} ± {numero(cv['desviacion'], 4)}."
    )
    tabla = pd.DataFrame({"Entrenamiento": art["metricas_entrenamiento"], "Prueba": art["metricas_prueba"]})
    tabla = tabla.rename(index={"R2": "R²"}).map(lambda v: numero(v, 4))
    st.dataframe(tabla, width="content")
    mostrar_figura(f"{ej['prefijo']}_evaluacion.png",
                   "Izquierda: valores reales vs. predichos. Derecha: residuos (deben repartirse sin patrón alrededor de 0).")

with tab_coef:
    terminos = " ".join(
        f"{'+' if c >= 0 else '−'} {numero(abs(c), 4)}·{v}" for v, c in art["coeficientes"].items()
    )
    st.markdown("**Ecuación del modelo**")
    st.code(f"{art['objetivo']} = {numero(art['intercepto'], 4)} {terminos}", language=None)

    beta = art["coef_estandarizados"]
    total = sum(abs(b) for b in beta.values())
    orden = sorted(art["variables"], key=lambda v: abs(beta[v]), reverse=True)
    filas, interpretaciones = [], []
    for variable in orden:
        campo = ej["campos"][variable]
        efecto = art["coeficientes"][variable] * campo["paso"]
        p = art["p_valores"][variable]
        filas.append({
            "Variable": variable,
            "Coeficiente": numero(art["coeficientes"][variable], 4),
            "β estandarizado": numero(beta[variable], 3),
            "Importancia": f"{numero(100 * abs(beta[variable]) / total, 1)} %",
            "p-valor": "< 0,001" if p < 0.001 else numero(p, 3),
            "¿Significativa?": "Sí" if p < 0.05 else "No",
        })
        interpretaciones.append(
            f"- **{variable}:** con {campo['texto_paso']}, la predicción "
            f"**{'sube' if efecto >= 0 else 'baja'} {numero(abs(efecto), 2)} {unidad}**, "
            f"manteniendo fijas las demás variables."
        )
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
    st.markdown("**Interpretación**\n\n" + "\n".join(interpretaciones))
    st.success(f"**Variable con mayor impacto:** {orden[0]} (β estandarizado = {numero(beta[orden[0]], 3)}).")
    st.caption(
        "El coeficiente estandarizado (β) mide el efecto en desviaciones estándar, así que permite comparar "
        "variables con unidades distintas. Una variable es significativa si su p-valor es menor que 0,05."
    )
    mostrar_figura(f"{ej['prefijo']}_importancia.png", "Verde: aumenta la predicción · Rojo: la disminuye.", ancho=640)

with tab_graficas:
    mostrar_figura(f"{ej['prefijo']}_dispersion.png",
                   "Relación de cada variable independiente con la variable dependiente (línea roja: ajuste lineal simple).")
    if ej["prefijo"] == "energia":
        mostrar_figura("energia_promedios.png",
                       "Consumo medio por hora y por día: crece casi en línea recta con la hora.")
    g1, g2 = st.columns(2)
    with g1:
        mostrar_figura(f"{ej['prefijo']}_correlacion.png", "Matriz de correlación de Pearson.")
    with g2:
        mostrar_figura(f"{ej['prefijo']}_cajas.png", "Diagramas de caja: distribución y valores atípicos.")

with tab_datos:
    st.write(f"**{ej['archivo']}** · {numero(len(datos), 0)} filas · {datos.shape[1]} columnas")
    st.dataframe(datos.describe().T, width="stretch")
    st.dataframe(datos.head(50), width="stretch", height=300)
