# dashboard.py
# RAMA diario: Analítica y Visualización (tu dataset)
# - Carga automática del CSV (misma carpeta)
# - Etiqueta "estación del año"
# - Ventana móvil, correlación, disimilitud, MDS, PCA
# - Lags (correlación cruzada)
# - Estacionalidad mensual, tendencia anual, análisis por estación

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


# =========================
# 1) Cargar CSV (simple)
# =========================
# Esto busca el archivo "rama_2023_05.csv" en la misma carpeta donde está dashboard.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "rama_2023_05.csv")


@st.cache_data(show_spinner=False)
def cargar_datos(csv_path):
    df = pd.read_csv(csv_path)

    # Convertir fecha
    if "fecha" not in df.columns:
        raise ValueError("No encuentro la columna 'fecha' en el CSV.")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    # Convertir contaminantes a numérico
    for c in df.columns:
        if c != "fecha":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def estacion_del_anio(fecha):
    # México/hemisferio norte (meteorológico)
    m = int(fecha.month)
    if m in (12, 1, 2):
        return "Invierno"
    if m in (3, 4, 5):
        return "Primavera"
    if m in (6, 7, 8):
        return "Verano"
    return "Otoño"


ORDEN_ESTACIONES = ["Invierno", "Primavera", "Verano", "Otoño"]


def suavizado_ventana(df, cols, window, method):
    x = df[["fecha"] + cols].copy().set_index("fecha")
    if method == "mean":
        x = x.rolling(window=window, min_periods=max(1, window // 2)).mean()
    else:
        x = x.rolling(window=window, min_periods=max(1, window // 2)).median()
    return x.reset_index()


def matriz_distancias(X_time_by_var, metric):
    # X_time_by_var: index=time, columns=variables
    A = X_time_by_var.to_numpy().T  # variables x tiempo
    D = squareform(pdist(A, metric=metric))
    return pd.DataFrame(D, index=X_time_by_var.columns, columns=X_time_by_var.columns)


def correr_mds(D):
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0,
        n_init=4,
        max_iter=300,
    )
    Z = mds.fit_transform(D.to_numpy())
    out = pd.DataFrame(Z, columns=["MDS1", "MDS2"])
    out["variable"] = D.index.tolist()
    return out


def pca_contaminantes(X_time_by_var):
    # PCA sobre variables (cada contaminante descrito por su serie temporal)
    A = X_time_by_var.to_numpy().T  # vars x tiempo
    A = StandardScaler().fit_transform(A)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(A)
    out = pd.DataFrame(Z, columns=["PC1", "PC2"])
    out["variable"] = X_time_by_var.columns.tolist()
    return out, pca.explained_variance_ratio_


def pca_dias_coloreado(df_f, cols):
    # PCA sobre días (cada día es un punto con varias variables)
    X = df_f[cols].dropna()
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xz)

    out = pd.DataFrame(Z, columns=["PC1", "PC2"])
    # Mantener etiquetas alineadas (mismo índice)
    out["estacion"] = df_f.loc[X.index, "estacion"].values
    return out, pca.explained_variance_ratio_


def correlacion_cruzada(a, b, max_lag):
    # correlación con desfase: lag>0 significa que A "va antes" que B
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)

    n = len(a)
    lags = range(-max_lag, max_lag + 1)
    vals = []
    for lag in lags:
        if lag < 0:
            x = a[-lag:]
            y = b[: n + lag]
        elif lag > 0:
            x = a[: n - lag]
            y = b[lag:]
        else:
            x = a
            y = b

        if len(x) < 20:
            vals.append(np.nan)
        else:
            vals.append(float(np.corrcoef(x, y)[0, 1]))

    return pd.DataFrame({"lag": list(lags), "corr": vals})


# =========================
# UI
# =========================
st.set_page_config(page_title="RAMA – Analítica y Visualización", layout="wide")
st.title("RAMA diario: Analítica y Visualización (tu dataset)")

if not os.path.exists(CSV_PATH):
    st.error(
        "No encuentro el archivo **rama_2023_05.csv** en la misma carpeta del script.\n\n"
        "Solución:\n"
        "1) Sube rama_2023_05.csv al repo (misma carpeta que dashboard.py)\n"
        "2) Asegúrate que se llame EXACTO: rama_2023_05.csv"
    )
    st.stop()

df = cargar_datos(CSV_PATH)

# Crear etiquetas
df["mes"] = df["fecha"].dt.month
df["anio"] = df["fecha"].dt.year
df["estacion"] = df["fecha"].apply(estacion_del_anio)

vars_all = [c for c in df.columns if c not in ["fecha", "mes", "anio", "estacion"]]

# Sidebar
st.sidebar.header("Controles")

dmin = df["fecha"].min().date()
dmax = df["fecha"].max().date()
d0, d1 = st.sidebar.date_input("Rango de fechas", value=(dmin, dmax), min_value=dmin, max_value=dmax)
d0 = pd.to_datetime(d0)
d1 = pd.to_datetime(d1)

selected = st.sidebar.multiselect("Variables a incluir", vars_all, default=vars_all)
if len(selected) < 2:
    st.warning("Selecciona mínimo 2 variables.")
    st.stop()

sel_est = st.sidebar.multiselect("Filtrar por estación", ORDEN_ESTACIONES, default=ORDEN_ESTACIONES)

window = st.sidebar.slider("Ventana móvil (días)", 1, 60, 7, 1)
smooth_method = st.sidebar.selectbox("Suavizado", ["mean", "median"], index=0)
metric = st.sidebar.selectbox("Disimilitud (para MDS)", ["correlation", "euclidean", "cosine"], index=0)

# Filtrado principal
df_f = df[(df["fecha"] >= d0) & (df["fecha"] <= d1)].copy()
df_f = df_f[df_f["estacion"].isin(sel_est)].copy()

if df_f.empty:
    st.error("Tu filtro dejó el dataset vacío (rango/estación).")
    st.stop()

tabs = st.tabs([
    "Dataset",
    "Series + Ventana",
    "Correlación",
    "MDS",
    "PCA",
    "PCA (días por estación)",
    "Secuencia (lags)",
    "Estacionalidad mensual",
    "Tendencia anual",
    "Por estación"
])

# =========================
# TAB: Dataset
# =========================
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{len(df_f):,}")
    c2.metric("Variables", f"{len(selected)}")
    c3.metric("Rango", f"{df_f['fecha'].min().date()} → {df_f['fecha'].max().date()}")
    c4.metric("Estaciones", ", ".join(sel_est) if sel_est else "—")

    st.subheader("Vista previa")
    st.dataframe(df_f[["fecha", "estacion"] + selected].head(15), use_container_width=True)

    st.subheader("Faltantes")
    na = df_f[selected].isna().sum()
    if int(na.sum()) == 0:
        st.success("✅ No hay faltantes en el filtro actual.")
    else:
        st.warning(f"⚠️ Hay {int(na.sum())} faltantes.")
        fig = px.bar(x=na.index, y=na.values, labels={"x": "Variable", "y": "Faltantes"})
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB: Series + Ventana
# =========================
with tabs[1]:
    st.subheader("Series temporales (suavizadas)")
    show_cols = st.multiselect("Variables a graficar", selected, default=selected[: min(4, len(selected))])
    if not show_cols:
        st.warning("Selecciona al menos 1 variable para graficar.")
    else:
        smooth = suavizado_ventana(df_f, show_cols, window=window, method=smooth_method)
        long = smooth.melt(id_vars=["fecha"], var_name="variable", value_name="value")
        fig = px.line(long, x="fecha", y="value", color="variable",
                      title=f"Suavizado {smooth_method} (ventana={window} días)")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB: Correlación
# =========================
with tabs[2]:
    st.subheader("Correlación entre contaminantes")
    corr = df_f[selected].corr()
    fig = px.imshow(corr, aspect="auto", zmin=-1, zmax=1, title="Matriz de correlación")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Prepro común (Z-score) para MDS/PCA de contaminantes
# =========================
X = df_f[["fecha"] + selected].set_index("fecha").dropna()
Xz = (X - X.mean(axis=0)) / (X.std(axis=0).replace(0, np.nan))
Xz = Xz.dropna(axis=0, how="any")

# =========================
# TAB: MDS
# =========================
with tabs[3]:
    st.subheader("MDS 2D: mapa de similitud entre contaminantes")
    if len(Xz) < 30:
        st.error("Muy pocos datos para MDS con ese filtro. Amplía rango/estaciones.")
    else:
        D = matriz_distancias(Xz, metric=metric)
        Z = correr_mds(D)
        fig = px.scatter(Z, x="MDS1", y="MDS2", text="variable", hover_name="variable",
                         title=f"MDS usando disimilitud: {metric} (con Z-score)")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB: PCA (contaminantes)
# =========================
with tabs[4]:
    st.subheader("PCA: componentes principales (comparando contaminantes)")
    if len(Xz) < 30:
        st.error("Muy pocos datos para PCA con ese filtro.")
    else:
        pca_df, evr = pca_contaminantes(Xz)
        fig1 = px.scatter(pca_df, x="PC1", y="PC2", text="variable", hover_name="variable",
                          title="PCA (PC1 vs PC2) sobre contaminantes (Z-score)")
        fig1.update_traces(textposition="top center")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(x=["PC1", "PC2"], y=evr, labels={"x": "Componente", "y": "Varianza explicada"},
                      title="Varianza explicada")
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB: PCA (días coloreado por estación)
# =========================
with tabs[5]:
    st.subheader("PCA de días (cada día es un punto) coloreado por estación")
    if len(df_f) < 50:
        st.error("Muy pocos días en el filtro para PCA de días.")
    else:
        pca_days, evr_days = pca_dias_coloreado(df_f, selected)
        fig = px.scatter(pca_days, x="PC1", y="PC2", color="estacion",
                         title="PCA de días (Z-score) — color = estación")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(x=["PC1", "PC2"], y=evr_days,
                      labels={"x": "Componente", "y": "Varianza explicada"},
                      title="Varianza explicada (PCA de días)")
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB: Secuencia (lags)
# =========================
with tabs[6]:
    st.subheader("Relaciones de secuencia: correlación con desfase (lags)")
    c1, c2 = st.columns(2)
    with c1:
        a_name = st.selectbox("Variable A", selected, index=0)
    with c2:
        b_name = st.selectbox("Variable B", selected, index=min(1, len(selected) - 1))

    max_lag = st.slider("Max lag (días)", 1, 180, 60, 1)

    temp = df_f[["fecha", a_name, b_name]].dropna()
    if len(temp) < 80:
        st.error("Muy pocos datos para lags con ese filtro. Amplía rango/estaciones.")
    else:
        cc = correlacion_cruzada(temp[a_name].to_numpy(), temp[b_name].to_numpy(), max_lag=max_lag)
        fig = px.line(cc, x="lag", y="corr", title=f"Correlación cruzada: {a_name} vs {b_name}")
        fig.add_hline(y=0)
        st.plotly_chart(fig, use_container_width=True)

        best = cc.loc[cc["corr"].abs().idxmax()]
        st.success(f"Lag con |corr| máxima: **{int(best['lag'])} días** (corr = {best['corr']:.3f}).")

# =========================
# TAB: Estacionalidad mensual
# =========================
with tabs[7]:
    st.subheader("Estacionalidad mensual (promedio por mes)")
    monthly = df_f.groupby("mes")[selected].mean().reset_index()
    long = monthly.melt(id_vars=["mes"], var_name="variable", value_name="value")
    fig = px.line(long, x="mes", y="value", color="variable", markers=True, title="Promedio por mes")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB: Tendencia anual
# =========================
with tabs[8]:
    st.subheader("Tendencia anual (promedio por año)")
    annual = df_f.groupby("anio")[selected].mean().reset_index()
    long = annual.melt(id_vars=["anio"], var_name="variable", value_name="value")
    fig = px.line(long, x="anio", y="value", color="variable", markers=True,
                  title="Promedio por año (2023 es parcial)")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB: Por estación
# =========================
with tabs[9]:
    st.subheader("Comparación por estación del año")

    # Promedio por estación
    seasonal = (
        df_f.groupby("estacion")[selected]
        .mean()
        .reindex(ORDEN_ESTACIONES)
        .reset_index()
    )
    long = seasonal.melt(id_vars=["estacion"], var_name="variable", value_name="value")
    fig = px.bar(long, x="estacion", y="value", color="variable", barmode="group",
                 title="Promedio por estación")
    st.plotly_chart(fig, use_container_width=True)

    # Boxplot por estación (distribución)
    long2 = df_f.melt(id_vars=["estacion"], value_vars=selected, var_name="variable", value_name="value")
    fig2 = px.box(long2, x="estacion", y="value", color="variable",
                  title="Distribución por estación (boxplot)")
    st.plotly_chart(fig2, use_container_width=True)
