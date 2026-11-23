"""Pruebas de la interfaz: cada ejercicio predice lo mismo que la ecuación del modelo."""

from pathlib import Path

import joblib
import pytest
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
APP = str(RAIZ / "app.py")

CASOS = {
    "Dólar": ("dolar", {"Dia": 120, "Inflacion": 0.02, "Tasa_interes": 5.0}),
    "Glucosa": ("glucosa", {"Edad": 45, "IMC": 25.0, "Actividad_Fisica": 5}),
    "Energía": ("energia", {"Temperatura": 22.0, "Hora": 14, "Dia_Semana": 6}),
}


def ecuacion(artefacto, entrada):
    return artefacto["intercepto"] + sum(artefacto["coeficientes"][v] * x for v, x in entrada.items())


def numero(valor, decimales):
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


@pytest.mark.parametrize("ejercicio", list(CASOS))
def test_prediccion_coincide_con_la_ecuacion(ejercicio):
    prefijo, entrada = CASOS[ejercicio]
    artefacto = joblib.load(RAIZ / "modelos" / f"modelo_{prefijo}.joblib")

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio(key="ejercicio").set_value(ejercicio).run()
    for variable, valor in entrada.items():
        at.number_input(key=f"{prefijo}_{variable}").set_value(valor)
    at.button[0].click().run()

    assert not at.exception
    esperado = ecuacion(artefacto, entrada)
    decimales = 2 if prefijo == "dolar" else 1
    assert at.metric[0].value.startswith(numero(esperado, decimales))


def test_valores_fuera_de_rango_muestran_advertencia():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio(key="ejercicio").set_value("Energía").run()
    at.number_input(key="energia_Temperatura").set_value(55.0)
    at.button[0].click().run()
    assert any("fuera del rango" in w.value for w in at.warning)


@pytest.mark.parametrize("prefijo", ["dolar", "glucosa", "energia"])
def test_modelo_exportado_tiene_lo_necesario(prefijo):
    artefacto = joblib.load(RAIZ / "modelos" / f"modelo_{prefijo}.joblib")
    for clave in ("modelo", "variables", "coeficientes", "metricas_prueba", "rangos", "version_sklearn"):
        assert clave in artefacto
    assert 0 < artefacto["metricas_prueba"]["R2"] <= 1
