import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

base = os.path.dirname(os.path.abspath(__file__))
ruta = base + "/rama_2023_05.csv"

@st.cache_data(show_spinner=False)
def cargar_datos(ruta):
    df= pd.read_csv(ruta)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    #Contaminantes a numérico
    for c in df.columns:
        if c != "fecha":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def estaciones (fecha):
    mes = int(fecha.month)
    if mes in (12,1,2):
        return "Invierno"
    elif mes in (3,4,5):
        return "Primavera"
    elif mes in (6,7,8):
        return "Verano"
    else:
        return "Otoño"

estacionesord= ["Invierno", "Primavera", "Verano", "Otoño"]

def suavizar (df, columnas, window, method):
    x= df[["fecha"] + columnas].copy().set_index("fecha")
    if method== "mean":
        x= x.rolling(window=window, min_periods=max(1, window//2)).mean()
    else:
        x=x.rolling(window=window, min_periods=max(1, window//2)).median()
    return x.reset_index()

def matrizDistancia (series_tiempo, metrica):
    matriz= series_tiempo.to_numpy().T
    distancias= squareform(pdist(matriz, metric= metrica))
    nombres= series_tiempo.columns.tolist()
    MaDist= pd.DataFrame(distancias, index=nombres, columns=nombres)
    return MaDist


def funcionMDS(MaDist):
    mds= MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=4, 
             max_iter=300)
    Z= mds.fit_transform(MaDist.to_numpy())
    salida= pd.DataFrame(Z, columns=["MDS1", "MDS2"])
    salida["variable"] = MaDist.index.tolist()
    return salida

def pcaContaminantes(series_tiempo):
    #PCA sobre las variables
    matriz= series_tiempo.to_numpy().T
    matriz= StandardScaler().fit_transform(matriz)
    modeloPCA= PCA(n_components=2, random_state=0)
    coords= modeloPCA.fit_transform(matriz)
    salida= pd.DataFrame(coords, columns=["PC1", "PC2"])
    salida["variable"]= series_tiempo.columns.tolist()
    return salida, modeloPCA.explained_variance_ratio_

def pcaDiasEstacion(df_filtrado, columnas):
    #PCA sobre los días
    X= df_filtrado[columnas].dropna()
    Xz= StandardScaler().fit_transform(X)
    modeloPCA= PCA(n_components=2, random_state=0)
    coords= modeloPCA.fit_transform(Xz)
    salida= pd.DataFrame(coords, columns=["PC1", "PC2"])
    salida["estacion"] = df_filtrado.loc[X.index, "estacion"].values
    return salida, modeloPCA.explained_variance_ratio_

def correlacionCruzada(a,b, max_lag):
    a= np.asarray(a, float)
    b= np.asarray(b, float)
    a= (a-a.mean()) / (a.std()+1e-12)
    b= (b-b.mean()) / (b.std()+1e-12)
    n= len(a)
    lags = range(-max_lag, max_lag+1)
    valores= []
    for i in lags: 
        if i<0:
            x= a[-i:]
            y= b[:n+i]
        elif i>0:
            x= a[:n- i]
            y = b[i:]
        else:
            x=a
            y=b
        if len(x) < 20:
            valores.append(np.nan)
        else:
            valores.append(float(np.corrcoef(x, y)[0, 1]))
    return pd.DataFrame({"lag": list(lags), "corr": valores})

#Interfaz streamlit
st.set_page_config(page_title="Proyecto Analítica y Visualiación de Datos", layout="wide")
st.title("Red Automática de Monitoreo Atmosférico CDMX")

df= cargar_datos(ruta)
df["mes"]= df["fecha"].dt.month
df["anio"] = df["fecha"].dt.year
df["estacion"] = df["fecha"].apply(estaciones)
#print(df.head())
contaminantes = [c for c in df.columns if c not in ["fecha", "mes", "anio", "estacion"]]

#Sidebar
st.sidebar.header("Controles")
fmin = df["fecha"].min().date()
fmax = df["fecha"].max().date()
f0, f1 = st.sidebar.date_input("Rango de fechas", value=(fmin, fmax), min_value=fmin, max_value=fmax)
f0 = pd.to_datetime(f0)
f1 = pd.to_datetime(f1)

seleccionados= st.sidebar.multiselect("Variables a incluir", contaminantes, default=contaminantes)
if len(seleccionados)<2:
    print ("Selecciona mínimo 2 contaminantes")
    st.stop()

estSel = st.sidebar.multiselect("Filtrar por estación", estacionesord, default=estacionesord)
window = st.sidebar.slider("Ventana móvil por días", 1, 60, 7, 1)
metodo = st.sidebar.selectbox("Suavizado", ["mean", "median"], index=0)
metrica = "euclidean"

#Filtrado principal
df_f = df[(df["fecha"] >= f0) & (df["fecha"] <= f1)].copy()
df_f = df_f[df_f["estacion"].isin(estSel)].copy()

if df_f.empty is True:
    st.error("El filtro dejó el dataset vacío")
    st.stop()

divs = st.tabs(["Dataset", "Series de tiempo", "Correlación", "MDS", "PCA", "Secuencia", 
                "Resumen temporal", "Comparación por estación"])

