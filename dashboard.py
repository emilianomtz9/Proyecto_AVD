#Martínez Salinas Emiliano
#Huerta Rodríguez Sofía
#5AM1

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

seleccionados= st.sidebar.multiselect("Contaminantes", contaminantes, default=contaminantes)
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

#Tab 1: Dataset
with divs[0]:
    c1,c2,c3 = st.columns(3)
    c1.metric("Filas", f"{len(df_f):,}")
    c2.metric("Variables", f"{len(seleccionados)}")
    c3.metric("Rango", "2015 → 2023")
    st.subheader("Vista previa")
    st.dataframe(df_f[["fecha", "estacion"] + seleccionados].head(15), use_container_width=True)

#Tab 2: Serie
with divs[1]:
    st.subheader("Series de tiempo")
    colMost = st.multiselect("Variables a graficar", seleccionados, default=seleccionados[:min(4,len(seleccionados))])
    if not colMost:
        st.warning("Escoge por lo menos una variable para graficar")
    else:
        suavizado= suavizar(df_f, colMost, window=window, method=metodo)
        largo =suavizado.melt(id_vars=["fecha"], var_name="contaminante", value_name="valor")
        fig =px.line(largo, x="fecha", y="valor", color="contaminante",
                      title=f"Suavizado usando {metodo} para {window} días")
        st.plotly_chart(fig, use_container_width=True)

#Tab 3: Correlación
with divs[2]:
    st.subheader("Correlación entre contaminantes")
    corr = df_f[seleccionados].corr()
    fig = px.imshow(corr,aspect="auto",zmin=-1, zmax=1 ,text_auto=".2f")
    fig.update_layout(title="Matriz de correlación",coloraxis_colorbar=dict(title="r", tickvals=[-1, -0.5, 0, 0.5, 1]))
    st.plotly_chart(fig, use_container_width=True)
    fig.update_yaxes(autorange="reversed")

#Normalización con z
x= df_f[["fecha"]+seleccionados].set_index("fecha").dropna()
xz= (x-x.mean(axis=0)) / (x.std(axis=0).replace(0,np.nan))
xz= xz.dropna(axis=0, how="any")

#Tab 4: MDS
with divs[3]:
    st.subheader("MDS entre contaminantes")
    if len(xz)<30:
        st.error("Muy pocos datos para ocupar MDS")
    else:
        matriz= matrizDistancia(xz, metrica)
        z = funcionMDS(matriz)
        fig = px.scatter(z, x="MDS1", y="MDS2", text="variable", hover_name="variable", 
                         title=f"MDS con estandarización ")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

#Tab 5: PCA
with divs[4]:
    st.subheader("PCA entre contaminantes")
    if len(xz)< 30:
        st.error("Muy pocos datos para ocupar PCA")
    else:
        pca_df, evr = pcaContaminantes(xz)
        fig1 = px.scatter(pca_df, x="PC1", y="PC2", text="variable", hover_name="variable",
                          title="PCA sobre contaminantes con estandarización")
        fig1.update_traces(textposition="top center")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(x=["PC1", "PC2"], y=evr, labels={"x": "Componente", "y": "Variabilidad"},
                      title="Variabilidad por componente")
        st.plotly_chart(fig2, use_container_width=True)

#Tab 6: Secuencias ************************
with divs[5]:
    st.subheader("Relaciones de secuencia con desfase")
    c1, c2= st.columns(2)
    with c1:
        a = st.selectbox("Variable 1", seleccionados, index=0)
    with c2:
        b = st.selectbox("Variable 2", seleccionados, index=min(1,len(seleccionados)-1))

    lagmax= st.slider("Lag máximo en días", 1,180,60,1)
    temp= df_f[["fecha", a, b]].dropna()
    if len(temp)< 80:
        st.error("Muy pocos datos para la secuencia de lags")
    else:
        cc = correlacionCruzada(temp[a].to_numpy(),temp[b].to_numpy(), max_lag=lagmax)
        fig = px.line(cc, x="lag", y="corr", title=f"Correlación cruzada entre {a} y {b}")
        fig.add_hline(y=0)
        st.plotly_chart(fig, use_container_width=True)
        mejor = cc.loc[cc["corr"].abs().idxmax()]
        st.success(f"Lag con correlación máxima: {int(mejor['lag'])} días (corr = {mejor['corr']:.3f})")

#Tab 7: Resumen temporal
with divs[6]:
    st.subheader("Resumen temporal")

    #Mensual
    st.markdown("Promedio histórico por mes")
    mensual = df_f.groupby("mes")[seleccionados].mean().reset_index()
    largom = mensual.melt(id_vars=["mes"], var_name="contaminante", value_name="valor")

    #min-max
    largom["valor_norm"] = largom.groupby("contaminante")["valor"].transform(
        lambda s: (s - s.min()) / (s.max()-s.min()+1e-12))
    figm_norm = px.line(largom, x="mes", y="valor_norm", color="contaminante",markers=True)
    st.plotly_chart(figm_norm, use_container_width=True)

    # Anual
    st.markdown("Promedio por año")
    anual =df_f.groupby("anio")[seleccionados].mean().reset_index()
    largoa =anual.melt(id_vars=["anio"], var_name="contaminante", value_name="valor")
    largoa["valor_norm"] = largoa.groupby("contaminante")["valor"].transform(
        lambda s: (s - s.min())/ (s.max()-s.min()+1e-12))

    figa_norm = px.line(largoa, x="anio", y="valor_norm", color="contaminante", markers=True)
    st.plotly_chart(figa_norm, use_container_width=True)

#Tab 8: Estadísticas por estación
with divs[7]:
    st.subheader("Comparación por estación del año")
    modo = st.radio("Escala", ["Real", "Normalizada"], horizontal=True)
    largo2 = df_f.melt(id_vars=["estacion"], value_vars=seleccionados,
                    var_name="variable", value_name="value")
    if modo =="Normalizada":
        largo2["value_plot"] = largo2.groupby("variable")["value"].transform(
            lambda s: (s-s.min()) /(s.max()-s.min()+1e-12))
    else:
        largo2["value_plot"] = largo2["value"]

    #Promedio por estación
    prom = largo2.groupby(["estacion","variable"])["value_plot"].mean().reset_index()
    prom["estacion"] =pd.Categorical(prom["estacion"], categories=estacionesord, ordered=True)
    fig =px.bar(prom, x="estacion", y="value_plot", color="variable", barmode="group",
                title=f"Promedio por estación")
    st.plotly_chart(fig, use_container_width=True)

    fig2 =px.box(largo2, x="estacion", y="value_plot", color="variable",
                title=f"Distribución por estación")
    st.plotly_chart(fig2, use_container_width=True)

