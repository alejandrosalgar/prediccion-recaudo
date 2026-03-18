"""
API mínima para el frontend Angular.
Misma lógica y fuente de datos que app.py (CSV).
Incluye endpoint RAG con Gemini para preguntas sobre los datos.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn import ensemble, linear_model, neighbors, svm, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

warnings.simplefilter("ignore")

DATA_URL = (
    "https://raw.githubusercontent.com/julianuribegomez/tesoreria/main/data13.csv"
)

app = Flask(__name__)

CORS(app)


# Cargar datos una vez al arrancar
_df = None


def get_df():
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_URL, sep=";")
    return _df


def get_models():
    return {
        "Regresión": linear_model.LinearRegression,
        "Arbol de decisión": tree.DecisionTreeRegressor,
        "k-NN": neighbors.KNeighborsRegressor,
        "Máquinas de soporte vectorial": svm.LinearSVR,
        "Gradient boosting": ensemble.GradientBoostingRegressor,
        "Random Forest": RandomForestRegressor,
        "MLPRegressor": lambda: MLPRegressor(max_iter=500),
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "BayesianRidge": BayesianRidge,
    }


@app.route("/api/datos/estadisticos", methods=["GET"])
def estadisticos():
    """Tabla estadísticos por periodo y mes (prom, desvest, máx, mín, cv)."""
    df = get_df()
    g = (
        df.groupby(["periodo", "mes"])["total_recaudo"]
        .agg([np.mean, np.std, np.max, np.min])
        .reset_index()
    )
    g = g.rename(
        columns={
            "periodo": "año",
            "mean": "prom",
            "std": "desvest",
            "amax": "máx",
            "amin": "mín",
        }
    )
    g = g.fillna(0)
    g = g[g["prom"] != 0]
    g["cv"] = g.apply(lambda row: row.desvest / row.prom if row.prom else 0, axis=1)
    return jsonify(g.round(0).to_dict("records"))


@app.route("/api/datos/variacion", methods=["GET"])
def variacion():
    """Tabla variación % mes a mes."""
    df = get_df()
    v = df.groupby(["periodo", "mes"])["total_recaudo"].sum().reset_index()
    v["%variacion"] = (
        (v["total_recaudo"] - v["total_recaudo"].shift(1))
        / v["total_recaudo"].shift(1)
        * 100
    )
    v["%variacion"] = v["%variacion"].fillna(0)  # primera fila sin anterior → 0
    return jsonify(v.round(0).to_dict("records"))


@app.route("/api/datos/metricas", methods=["GET"])
def metricas():
    """Tabla RMSE, MAE, R2 por modelo."""
    df = get_df()
    df_scatter = pd.DataFrame(
        df.groupby(["periodo", "mes"])[["total_recaudo", "total_transacciones"]].sum()
    ).reset_index()
    X_raw = df_scatter["total_transacciones"].values[:, None]
    y_raw = df_scatter["total_recaudo"].values
    X = np.log1p(X_raw)
    y = np.log1p(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    modelos_tabla = [
        ("Regresión", linear_model.LinearRegression()),
        ("Arbol de decisión", tree.DecisionTreeRegressor()),
        ("k-NN", neighbors.KNeighborsRegressor()),
        ("Máquinas de soporte vectorial", svm.LinearSVR()),
        ("Gradient boosting", ensemble.GradientBoostingRegressor()),
        ("Random Forest", RandomForestRegressor()),
        ("MLPRegressor", MLPRegressor(max_iter=500)),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
        ("ElasticNet", ElasticNet()),
        ("BayesianRidge", BayesianRidge()),
    ]
    out = []
    for nombre, modelo in modelos_tabla:
        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            out.append(
                {
                    "Modelo": nombre,
                    "RMSE": f"{rmse:,.0f}",
                    "MAE": f"{mae:,.0f}",
                    "R2": f"{r2:.3f}",
                }
            )
        except Exception:
            out.append(
                {"Modelo": nombre, "RMSE": "Error", "MAE": "Error", "R2": "Error"}
            )
    return jsonify(out)


@app.route("/api/graficas/scatter-transacciones", methods=["GET"])
def scatter_transacciones():
    """Datos para gráfica transacciones vs recaudo (log). Incluye curva de predicción si se pasa ?modelo=Nombre."""
    modelo_name = request.args.get("modelo")
    df = get_df()
    df_scatter = pd.DataFrame(
        df.groupby(["periodo", "mes"])[["total_recaudo", "total_transacciones"]].sum()
    ).reset_index()
    X_raw = df_scatter["total_transacciones"].values[:, None]
    y_raw = df_scatter["total_recaudo"].values
    X = np.log1p(X_raw.squeeze())
    y = np.log1p(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    out = {
        "entrenamiento": {"x": X_train.tolist(), "y": y_train.tolist()},
        "prueba": {"x": X_test.tolist(), "y": y_test.tolist()},
    }
    if modelo_name and modelo_name in get_models():
        model = get_models()[modelo_name]()
        model.fit(X_train.reshape(-1, 1), y_train)
        x_range = np.linspace(X.min(), X.max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        out["prediccion"] = {"x": x_range.tolist(), "y": y_pred.tolist()}
    return jsonify(out)


@app.route("/api/graficas/scatter-mes", methods=["GET"])
def scatter_mes():
    """Datos para gráfica mes (índice) vs recaudo (log). Incluye curva de predicción si se pasa ?modelo=Nombre."""
    modelo_name = request.args.get("modelo")
    df = get_df()
    periodo_group = (
        pd.DataFrame(df.groupby(["periodo", "mes"])["total_recaudo"].sum())
        .reset_index()
        .reset_index()
    )
    X = periodo_group["index"].values
    y_raw = periodo_group["total_recaudo"].values
    y = np.log1p(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    out = {
        "entrenamiento": {"x": X_train.tolist(), "y": y_train.tolist()},
        "prueba": {"x": X_test.tolist(), "y": y_test.tolist()},
    }
    if modelo_name and modelo_name in get_models():
        model = get_models()[modelo_name]()
        model.fit(X_train.reshape(-1, 1), y_train)
        x_range = np.linspace(X.min(), X.max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        out["prediccion"] = {"x": x_range.tolist(), "y": y_pred.tolist()}
    return jsonify(out)


@app.route("/api/prediccion/transacciones", methods=["GET"])
def prediccion_transacciones():
    """Predicción de recaudo dado número de transacciones. Query: modelo, valor."""
    modelo_name = request.args.get("modelo", "Arbol de decisión")
    try:
        valor = float(request.args.get("valor", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "Ingrese un número válido."}), 400
    if valor == 0:
        return jsonify({"error": "Ingrese un número diferente de cero."}), 400

    df = get_df()
    df_scatter = pd.DataFrame(
        df.groupby(["periodo", "mes"])[["total_recaudo", "total_transacciones"]].sum()
    ).reset_index()
    X = df_scatter["total_transacciones"].values[:, None]
    y = df_scatter["total_recaudo"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    models = get_models()
    if modelo_name not in models:
        return jsonify({"error": "Modelo no válido."}), 400
    model = models[modelo_name]()
    model.fit(X_train, y_train)
    pred = model.predict([[valor]])
    return jsonify({"prediccion": float(pred[0])})


@app.route("/api/prediccion/mes", methods=["GET"])
def prediccion_mes():
    """Predicción de recaudo dado mes (índice). Query: modelo, valor."""
    modelo_name = request.args.get("modelo", "Arbol de decisión")
    try:
        valor = float(request.args.get("valor", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "Ingrese un número válido."}), 400
    if valor == 0:
        return jsonify({"error": "Ingrese un número diferente de cero."}), 400

    df = get_df()
    periodo_group = (
        pd.DataFrame(df.groupby(["periodo", "mes"])["total_recaudo"].sum())
        .reset_index()
        .reset_index()
    )
    X = periodo_group["index"].values[:, None]
    y = periodo_group["total_recaudo"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    models = get_models()
    if modelo_name not in models:
        return jsonify({"error": "Modelo no válido."}), 400
    model = models[modelo_name]()
    model.fit(X_train, y_train)
    pred = model.predict([[valor]])
    return jsonify({"prediccion": float(pred[0])})


@app.route("/api/modelos", methods=["GET"])
def listar_modelos():
    """Lista de nombres de modelos disponibles."""
    return jsonify(list(get_models().keys()))


# --- RAG con Gemini ---
_global_summary_cache = None

# Respuestas directas desde los datos (sin llamar a Gemini) para evitar 429
_aggregates_cache = None


def _fmt_recaudo(x) -> str:
    """
    Formatea un valor de recaudo como string sin notación científica,
    para que se muestre tal cual (ej. 9,749,359,000).
    """
    if pd.isna(x):
        return ""
    try:
        n = float(x)
        if abs(n) >= 1e6 or (abs(n) < 0.01 and n != 0):
            return f"{n:,.0f}"
        return f"{n:,.2f}" if abs(n) != int(n) else f"{int(n):,}"
    except (TypeError, ValueError):
        return str(x)


def _get_aggregates():
    """Precalcula totales por año y por (contrato, año). Cacheado."""
    global _aggregates_cache
    if _aggregates_cache is not None:
        return _aggregates_cache
    df = get_df()
    por_año = (
        df.groupby("periodo")["total_recaudo"]
        .sum()
        .round(0)
        .to_dict()
    )
    por_contrato_año = (
        df.groupby(["contrato", "periodo"])["total_recaudo"]
        .sum()
        .round(0)
        .reset_index()
    )
    _aggregates_cache = {
        "por_año": por_año,
        "por_contrato_año": por_contrato_año,
        "años_disponibles": sorted(por_año.keys()),
    }
    return _aggregates_cache


def _normalize_year_from_question(question_lower: str) -> list:
    """Extrae años de la pregunta; interpreta 'año pasado', 'este año', etc."""
    from datetime import date
    current_year = date.today().year
    years = set()
    # Años explícitos (4 dígitos)
    for token in question_lower.replace(",", " ").replace(".", " ").split():
        if token.isdigit() and len(token) == 4:
            try:
                years.add(int(token))
            except ValueError:
                pass
    # Expresiones relativas
    if "año pasado" in question_lower or "ano pasado" in question_lower or "último año" in question_lower:
        years.add(current_year - 1)
    if "este año" in question_lower or "año actual" in question_lower:
        years.add(current_year)
    if "2020" in question_lower:
        years.add(2020)
    if "2021" in question_lower:
        years.add(2021)
    if "2022" in question_lower:
        years.add(2022)
    if "2023" in question_lower:
        years.add(2023)
    if "2024" in question_lower:
        years.add(2024)
    if "2025" in question_lower:
        years.add(2025)
    agg = _get_aggregates()
    return [y for y in sorted(years) if y in agg["años_disponibles"]]


def _contratos_from_question(question_lower: str, df):
    """Contratos cuyo nombre aparece en la pregunta."""
    contratos_unicos = df["contrato"].dropna().unique().tolist()
    return [c for c in contratos_unicos if str(c).lower() in question_lower]


_MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


def _mes_from_question(question_lower: str):
    """Extrae el mes de la pregunta (nombre o «mes X»). Devuelve int o None."""
    for nombre, num in _MESES.items():
        if nombre in question_lower:
            return num
    import re
    # "mes 2", "mes 02", "en el mes 3", etc.; no confundir con años de 4 dígitos
    match = re.search(r"\bmes\s+(\d{1,2})\b", question_lower)
    if match:
        m = int(match.group(1))
        if 1 <= m <= 12:
            return m
    return None


def _try_direct_answer(question: str):
    """
    Responde con los datos directamente cuando la pregunta es tipo
    «recaudo total en X», «recaudo en [contrato] en [año]», «qué tiene esta bd», etc.
    Devuelve la respuesta en texto o None si hay que usar Gemini.
    """
    q = question.strip().lower()
    df = get_df()
    agg = _get_aggregates()
    por_año = agg["por_año"]
    por_contrato_año = agg["por_contrato_año"]
    años_disp = agg["años_disponibles"]

    # Preguntas sobre qué hay en la base / qué datos tiene (sin llamar a Gemini)
    if any(
        t in q
        for t in (
            "qué tiene", "que tiene", "qué hay", "que hay", "qué datos",
            "que datos", "describe", "de qué trata", "de que trata",
            "qué contiene", "que contiene", "información de la bd",
            "qué información", "resumen del dato", "qué es esta bd",
        )
    ):
        n_filas = len(df)
        años_str = f"{int(min(años_disp))} a {int(max(años_disp))}"
        total = df["total_recaudo"].sum()
        n_contratos = df["contrato"].nunique()
        return (
            f"Esta base tiene datos de recaudo de la tesorería: {n_filas:,} registros, "
            f"desde {años_str}. Incluye recaudo por concepto (ej. impuesto predial, industria y comercio), "
            f"por mes y por tipo de recaudo (presencial/virtual). "
            f"En total hay {n_contratos} conceptos distintos y el recaudo total histórico es del orden de {_fmt_recaudo(total)} pesos. "
            f"Puedes preguntar por un año o concepto concreto, o pedir una gráfica (por ejemplo: «Gráficame el recaudo por año»)."
        )

    años = _normalize_year_from_question(q)
    contratos = _contratos_from_question(q, df)
    mes_num = _mes_from_question(q)

    # ¿Cuánto se recaudó en [mes] [año]? / recaudo en febrero 2020
    if mes_num is not None and años and ("recaudo" in q or "recaudó" in q or "cuánto" in q or "cuanto" in q):
        año = años[0] if len(años) == 1 else max(años)
        filas = df[(df["periodo"] == año) & (df["mes"] == mes_num)]
        if not filas.empty:
            total = filas["total_recaudo"].sum()
            nombre_mes = [k for k, v in _MESES.items() if v == mes_num][0].capitalize()
            return f"En {nombre_mes} de {int(año)} el recaudo total fue de {_fmt_recaudo(total)} pesos."

    # ¿Cuánto se recaudó en total en [año]? / recaudo total en [año]
    if años and (
        "total" in q or "recaudo" in q and "cuánto" in q or "cuanto" in q
    ):
        if len(años) == 1:
            valor = por_año.get(años[0])
            if valor is not None:
                return f"En {años[0]} el recaudo total fue de {_fmt_recaudo(valor)} pesos (suma de todos los conceptos y meses)."
        else:
            partes = [f"En {y}: {_fmt_recaudo(por_año.get(y, 0))} pesos" for y in años]
            return "Recaudo total por año: " + "; ".join(partes) + "."

    # ¿Cuánto en [contrato] en [año]? / recaudo en impuesto predial año pasado
    if contratos and (años or "año" in q or "ano" in q):
        años_use = años if años else agg["años_disponibles"][-3:]  # últimos 3 años
        respuestas = []
        for c in contratos[:3]:  # máx 3 contratos
            filas = por_contrato_año[
                (por_contrato_año["contrato"] == c) & (por_contrato_año["periodo"].isin(años_use))
            ]
            if filas.empty:
                continue
            total = filas["total_recaudo"].sum()
            años_str = ", ".join(str(int(y)) for y in sorted(filas["periodo"].unique()))
            respuestas.append(
                f"En {c} ({años_str}): {_fmt_recaudo(total)} pesos."
            )
        if respuestas:
            return " ".join(respuestas)
        if contratos and not años:
            # Solo contrato, sin año: dar último año disponible
            ultimo_año = max(agg["años_disponibles"])
            filas = por_contrato_año[
                (por_contrato_año["contrato"] == contratos[0])
                & (por_contrato_año["periodo"] == ultimo_año)
            ]
            if not filas.empty:
                total = filas["total_recaudo"].sum()
                return f"En {ultimo_año}, el recaudo en «{contratos[0]}» fue de {_fmt_recaudo(total)} pesos."
    return None


def _wants_chart(question_lower: str) -> bool:
    """Detecta si la pregunta pide una gráfica."""
    triggers = (
        "gráficame", "graficame", "gráfica", "grafica", "gráfico", "grafico",
        "muéstrame un gráfico", "muestrame un grafico", "muéstrame una gráfica",
        "dame una gráfica", "dame un gráfico", "visualiza", "ver una gráfica",
        "mostrar gráfica", "quiero ver", "pintar", "dibuja",
    )
    return any(t in question_lower for t in triggers)


def _build_chart_data(question: str):
    """
    Si la pregunta pide una gráfica, devuelve datos para mostrarla.
    Retorna dict con type, title, labels, values o None.
    """
    q = question.strip().lower()
    if not _wants_chart(q):
        return None
    df = get_df()
    agg = _get_aggregates()
    por_año = agg["por_año"]
    por_contrato_año = agg["por_contrato_año"]
    años_disp = agg["años_disponibles"]
    años = _normalize_year_from_question(q)

    # "Gràficame recaudo por año" / "recaudo total por año"
    if "año" in q or "ano" in q or "periodo" in q or "años" in q or not años:
        # Gráfica recaudo por año (últimos 12 años para no saturar)
        años_show = sorted(años_disp)[-12:] if len(años_disp) > 12 else sorted(años_disp)
        labels = [str(int(y)) for y in años_show]
        values = [float(por_año.get(y, 0)) for y in años_show]
        if not values or max(values) == 0:
            return None
        return {
            "type": "bar",
            "title": "Recaudo total por año (pesos)",
            "labels": labels,
            "values": values,
        }

    # "Gráficame recaudo por mes en 2024" / "recaudo por mes este año"
    if años and ("mes" in q or "mensual" in q):
        año = años[-1]
        df_a = df[df["periodo"] == año]
        if df_a.empty:
            return None
        por_mes = df_a.groupby("mes")["total_recaudo"].sum().round(0)
        meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        labels = [meses_nombres[int(m) - 1] if 1 <= m <= 12 else str(m) for m in sorted(por_mes.index)]
        values = [float(por_mes[m]) for m in sorted(por_mes.index)]
        return {
            "type": "bar",
            "title": f"Recaudo por mes en {int(año)} (pesos)",
            "labels": labels,
            "values": values,
        }

    # "Gráficame top contratos" / "principales conceptos"
    if "contrato" in q or "concepto" in q or "principal" in q or "top" in q or "mayor" in q:
        año_use = int(años[0]) if años else max(años_disp)
        filas = por_contrato_año[por_contrato_año["periodo"] == año_use]
        if filas.empty:
            filas = por_contrato_año[por_contrato_año["periodo"] == max(años_disp)]
        top = filas.nlargest(10, "total_recaudo")
        labels = [str(c)[:25] for c in top["contrato"].tolist()]
        values = top["total_recaudo"].tolist()
        if not values:
            return None
        return {
            "type": "bar",
            "title": f"Top 10 conceptos por recaudo en {año_use} (pesos)",
            "labels": labels,
            "values": values,
        }

    # Por defecto: recaudo por año (últimos años)
    años_show = sorted(años_disp)[-12:] if len(años_disp) > 12 else sorted(años_disp)
    labels = [str(int(y)) for y in años_show]
    values = [float(por_año.get(y, 0)) for y in años_show]
    if not values or max(values) == 0:
        return None
    return {
        "type": "bar",
        "title": "Recaudo total por año (pesos)",
        "labels": labels,
        "values": values,
    }


def _build_global_summary():
    """Resumen global breve del dataset para dar contexto general al LLM (cacheado)."""
    global _global_summary_cache
    if _global_summary_cache is not None:
        return _global_summary_cache
    df = get_df()
    lines = []
    lines.append(
        "Resumen global del dataset de recaudo (columnas clave: contrato, tipo_archivo_recaudo, periodo, mes, total_recaudo, total_transacciones)."
    )
    lines.append(f"- Número de filas: {len(df)}")
    lines.append(f"- Periodos (años): {df['periodo'].min():.0f} a {df['periodo'].max():.0f}")
    lines.append(f"- Total recaudo (suma): {_fmt_recaudo(df['total_recaudo'].sum())}")
    lines.append(
        f"- Promedio total_recaudo por fila: {_fmt_recaudo(df['total_recaudo'].mean())}"
    )
    resumen_periodo = (
        df.groupby("periodo")["total_recaudo"]
        .agg(["sum", "mean", "count"])
        .round(0)
    )
    resumen_periodo_str = resumen_periodo.copy()
    resumen_periodo_str["sum"] = resumen_periodo_str["sum"].apply(_fmt_recaudo)
    resumen_periodo_str["mean"] = resumen_periodo_str["mean"].apply(_fmt_recaudo)
    lines.append("\nRecaudo agregado por periodo (año):")
    lines.append(resumen_periodo_str.to_string())
    _global_summary_cache = "\n".join(lines)
    return _global_summary_cache


def _build_rag_context(question: str) -> str:
    """
    Construye un contexto reducido y específico para la pregunta.

    Estrategia muy ligera de “retrieval”:
    - Detecta años mencionados en la pregunta.
    - Detecta contratos mencionados (por coincidencia de texto).
    - Filtra el DataFrame por esos criterios si existen.
    - Toma solo una muestra pequeña de filas relevantes.
    """
    df = get_df()
    q_lower = question.lower()

    # Extraer posibles años mencionados en la pregunta (entre min y max del dataset)
    years = set()
    for token in q_lower.replace(",", " ").replace(".", " ").split():
        if token.isdigit() and len(token) == 4:
            try:
                y = int(token)
            except ValueError:
                continue
            years.add(y)

    # Detectar contratos por coincidencia aproximada en el texto de la pregunta
    contratos_unicos = df["contrato"].dropna().unique().tolist()
    contratos_match = []
    for c in contratos_unicos:
        c_lower = str(c).lower()
        if c_lower in q_lower:
            contratos_match.append(c)

    df_filt = df
    if years:
        df_filt = df_filt[df_filt["periodo"].isin(years)]
    if contratos_match:
        df_filt = df_filt[df_filt["contrato"].isin(contratos_match)]

    # Si el filtro queda vacío, volver al dataset completo (pero solo muestrear)
    if df_filt.empty:
        df_filt = df

    # Muestra reducida para no saturar el prompt
    df_sample = df_filt.copy()
    df_sample = df_sample.sort_values(["periodo", "mes"]).head(40).round(0)

    lines = [_build_global_summary()]

    if years:
        lines.append(f"\nFiltro por años mencionados en la pregunta: {sorted(years)}")
    if contratos_match:
        lines.append(
            "\nFiltro por contratos mencionados en la pregunta: "
            + ", ".join(sorted(map(str, contratos_match)))
        )

    resumen_filtrado = (
        df_filt.groupby(["periodo", "mes"])["total_recaudo"]
        .agg(["sum", "mean", "count"])
        .round(0)
        .reset_index()
    )
    resumen_filtrado = resumen_filtrado.head(40).copy()
    resumen_filtrado["sum"] = resumen_filtrado["sum"].apply(_fmt_recaudo)
    resumen_filtrado["mean"] = resumen_filtrado["mean"].apply(_fmt_recaudo)
    lines.append(
        "\nResumen agregado del subconjunto relevante por periodo y mes (sum, mean, count de total_recaudo):"
    )
    lines.append(resumen_filtrado.to_string(index=False))

    for col in ["total_recaudo", "total_transacciones"]:
        if col in df_sample.columns:
            df_sample[col] = df_sample[col].apply(_fmt_recaudo)
    lines.append("\nMuestra de filas relevantes (máx 40):")
    lines.append(df_sample.to_string(index=False))

    return "\n".join(lines)


def _clean_scientific_in_answer(text: str) -> str:
    """Sustituye notación científica en la respuesta por cifra legible (ej. 9.74e+09 -> 9,749,359,000)."""
    import re
    if not text:
        return text
    pattern = re.compile(r"\b(\d+\.?\d*[eE][+-]?\d+)\b")
    def repl(m):
        try:
            return _fmt_recaudo(float(m.group(1)))
        except (ValueError, TypeError):
            return m.group(0)
    return pattern.sub(repl, text)


# Modelos a probar en orden (por si alguno devuelve 404)
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-pro"]


def _ask_gemini(api_key: str, context: str, question: str) -> str:
    """Llama a la API de Gemini con el contexto y la pregunta."""
    prompt = f"""Eres un asistente que responde preguntas sobre datos de recaudo de tesorería.
Usa ÚNICAMENTE la información del siguiente contexto para responder. Responde en español de forma clara y concisa.
Importante: cuando cites cifras de recaudo o cantidades, escríbelas siempre como número completo con separador de miles (ej. 9,749,359,000). Nunca uses notación científica (ej. 9.74e+09).

CONTEXTO (datos):
{context}

PREGUNTA DEL USUARIO:
{question}

RESPUESTA:"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }
    last_error = None
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        for attempt in range(3):  # hasta 3 intentos (1 inicial + 2 reintentos en 429)
            try:
                resp = requests.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    break
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    break
                return parts[0].get("text", "").strip()
            except requests.RequestException as e:
                last_error = e
                resp = getattr(e, "response", None)
                if resp is not None:
                    if resp.status_code == 404:
                        last_error = None
                        break
                    if resp.status_code == 429 and attempt < 2:
                        time.sleep(3 + attempt * 2)
                        continue
                raise
    if last_error:
        raise last_error
    return "No se obtuvo respuesta del modelo."


def _ask_groq(api_key: str, context: str, question: str) -> str:
    """
    Fallback opcional: llama a Groq (API gratuita con buenos límites).
    Usar cuando Gemini devuelve 429. Variable de entorno: GROQ_API_KEY.
    """
    prompt = (
        "Eres un asistente que responde preguntas sobre datos de recaudo de tesorería. "
        "Usa ÚNICAMENTE la información del siguiente contexto. Responde en español, claro y conciso. "
        "Cuando cites cifras de recaudo, escríbelas como número completo con separador de miles (ej. 9,749,359,000), nunca en notación científica.\n\n"
        f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}\n\nRESPUESTA:"
    )
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        return (choice.get("message", {}).get("content") or "").strip()
    except Exception:
        raise


# Cache de respuestas RAG (pregunta normalizada -> respuesta), máx 50 entradas
_rag_answer_cache = {}
_RAG_CACHE_MAX = 50


@app.route("/api/rag/ask", methods=["POST"])
def rag_ask():
    """RAG: responde preguntas sobre los datos usando Gemini."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return (
            jsonify(
                {
                    "error": "RAG no configurado: falta GEMINI_API_KEY. Añádela en las variables de entorno del servicio."
                }
            ),
            503,
        )
    body = request.get_json() or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Falta el campo 'question' en el cuerpo."}), 400

    # Respuesta cacheada para la misma pregunta (normalizada)
    key = question.lower().strip()
    chart_data = _build_chart_data(question)
    payload = {}
    if chart_data:
        payload["chart"] = chart_data

    if key in _rag_answer_cache:
        payload["answer"] = _rag_answer_cache[key]
        return jsonify(payload)

    # Intentar responder solo con los datos (sin Gemini) para evitar 429
    direct = _try_direct_answer(question)
    if direct is not None:
        direct = _clean_scientific_in_answer(direct)
        if len(_rag_answer_cache) >= _RAG_CACHE_MAX:
            _rag_answer_cache.pop(next(iter(_rag_answer_cache)))
        _rag_answer_cache[key] = direct
        payload["answer"] = direct
        return jsonify(payload)

    try:
        context = _build_rag_context(question)
        try:
            answer = _ask_gemini(api_key, context, question)
        except requests.RequestException as e:
            resp = getattr(e, "response", None)
            # Si Gemini devuelve 429, intentar fallback con Groq (gratuito, buenos límites)
            if resp is not None and resp.status_code == 429:
                groq_key = os.environ.get("GROQ_API_KEY", "").strip()
                if groq_key:
                    try:
                        answer = _ask_groq(groq_key, context, question)
                    except Exception:
                        return (
                            jsonify(
                                {
                                    "error": "Límite de consultas alcanzado. Espera un minuto o añade GROQ_API_KEY para usar un respaldo gratuito. Preguntas como «¿Cuánto se recaudó en 2024?» o «¿Qué tiene esta base?» se responden al instante sin límite."
                                }
                            ),
                            429,
                        )
                else:
                    return (
                        jsonify(
                            {
                                "error": "Límite de consultas alcanzado. Espera un minuto e inténtalo de nuevo. Para evitar esto: 1) Añade GROQ_API_KEY (gratis en groq.com) como respaldo, o 2) Activa facturación en Google AI Studio para más cuota con Gemini. Preguntas como «¿Qué tiene esta base?» o «¿Cuánto se recaudó en 2024?» se responden al instante."
                            }
                        ),
                        429,
                    )
            else:
                raise
        answer = _clean_scientific_in_answer(answer)
        if len(_rag_answer_cache) >= _RAG_CACHE_MAX:
            _rag_answer_cache.pop(next(iter(_rag_answer_cache)))
        _rag_answer_cache[key] = answer
        payload["answer"] = answer
        return jsonify(payload)
    except requests.RequestException as e:
        resp = getattr(e, "response", None)
        if resp is not None and resp.status_code == 429:
            return (
                jsonify(
                    {
                        "error": "Límite de consultas. Espera un minuto. Opciones: añade GROQ_API_KEY (gratis) o activa facturación en Gemini para más cuota."
                    }
                ),
                429,
            )
        return jsonify({"error": f"Error al llamar al servicio: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
