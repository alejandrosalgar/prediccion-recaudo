"""
API mínima para el frontend Angular.
Misma lógica y fuente de datos que app.py (CSV).
"""

import warnings

import numpy as np
import pandas as pd
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
