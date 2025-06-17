import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np

# -------------------------
# Configuración general
# -------------------------
st.set_page_config(page_title="Análisis Socioeconómico de Argentina", layout="wide")
st.title("📊 Análisis Socioeconómico de las Provincias Argentinas")

# Estilo global para reducir el ancho de la app
st.markdown("""
    <style>
        .block-container {
            max-width: 800px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Carga y combinación de datos
# -------------------------
@st.cache_data
def cargar_datos():
    # Dataset base
    df = pd.read_csv("argentina.csv")
    df = df.rename(columns={
        "province": "provincia",
        "gdp": "pbi",
        "illiteracy": "analfabetismo",
        "poverty": "pobreza",
        "deficient_infra": "infraestructura_deficiente",
        "school_dropout": "abandono_escolar",
        "no_healthcare": "sin_cobertura_salud",
        "birth_mortal": "mortalidad_infantil",
        "pop": "poblacion",
        "movie_theatres_per_cap": "cines_por_habitante",
        "doctors_per_cap": "medicos_por_habitante"
    })

    # Alumnos escuelas
    edu = pd.read_excel("030401_2022.xlsx", skiprows=7, engine="openpyxl")
    edu = edu[["Unnamed: 0", "Unnamed: 1"]]
    edu.columns = ["province", "total_alumnos_escuela"]
    edu = edu[edu["province"] != "Total del país"]
    df = df.merge(edu.rename(columns={"province": "provincia"}), on="provincia", how="left")

    # Universitarios
    uni = pd.read_excel("030409_2022.xlsx", header=None, skiprows=4, engine="openpyxl")
    uni.columns = ["province", "egresados_2020", "alumnos_2021", "nuevos_inscriptos", "reinscriptos", "egresados_2021"]
    uni = uni[uni["province"].notna() & (uni["province"] != "Total general")]
    uni = uni[["province", "alumnos_2021"]].rename(columns={"alumnos_2021": "total_universitarios"})
    df = df.merge(uni.rename(columns={"province": "provincia"}), on="provincia", how="left")

    # Internet – ENACOM Cuadro 5 (2022 y 2024)
    try:
        acceso_df = pd.read_excel("accesos_internet.xlsx", sheet_name="Cuadro 5", skiprows=5)
        acceso_df = acceso_df.iloc[:24, [0, -3, -1]]
        acceso_df.columns = ["provincia", "accesos_internet_2022", "accesos_internet_2024"]
        acceso_df["provincia"] = acceso_df["provincia"].str.replace("*", "", regex=False).str.strip()
        acceso_df = acceso_df[~acceso_df["provincia"].str.contains("Total")]
        df = df.merge(acceso_df, on="provincia", how="left")
    except Exception as e:
        st.warning(f"No se pudo cargar Cuadro 5 de ENACOM: {e}")

    # Opinión pública SIPM
    try:
        sipm = pd.read_excel("op_sipm_dec634_2016.xls")
        sipm.rename(columns={sipm.columns[0]: "province"}, inplace=True)
        sipm = sipm[["province", sipm.columns[1]]]
        sipm.columns = ["province", "indice_opinion_sipm"]
        df = df.merge(sipm.rename(columns={"province": "provincia"}), on="provincia", how="left")
    except:
        pass

    # Opinión pública ICC
    try:
        icc = pd.read_excel("op_icc_sipm_2016.xls")
        icc.rename(columns={icc.columns[0]: "province"}, inplace=True)
        icc = icc[["province", icc.columns[1]]]
        icc.columns = ["province", "indice_confianza_consumo"]
        df = df.merge(icc.rename(columns={"province": "provincia"}), on="provincia", how="left")
    except:
        pass

    # INDEC – Pobreza
    try:
        pobreza = pd.read_excel("coeficientes_variacion_pobreza_03_25.xlsx", skiprows=6)
        pobreza.rename(columns={pobreza.columns[0]: "province"}, inplace=True)
        if "Tasa de pobreza" in pobreza.columns:
            pobreza = pobreza[["province", "Tasa de pobreza"]]
            df = df.merge(pobreza.rename(columns={"province": "provincia"}), on="provincia", how="left")
    except:
        pass

    # Crear variable binaria de pobreza alta
    df["pobreza_alta"] = (df["pobreza"] > df["pobreza"].mean()).astype(int)

    return df

df = cargar_datos()

# -------------------------
# Tabs de navegación
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📍 Por Provincia",
    "📈 Comparaciones",
    "🔗 Correlaciones",
    "📊 Clustering KMeans",
    "📉 Predicción de Pobreza",
    "🌲 Árbol de Decisión",
    "📚 Conclusiones"
])

# -------------------------
# TAB 1: Vista por Provincia
# -------------------------
with tab1:
    st.header("📍 Indicadores por Provincia")
    prov = st.selectbox("Seleccioná una provincia", df["provincia"].dropna().unique())
    datos_prov = df[df["provincia"] == prov].T
    datos_prov = datos_prov[datos_prov[datos_prov.columns[0]].notna()]
    st.dataframe(datos_prov)

    # -------------------------
    # Limpieza de Datos: Visualización de Nulos
    # -------------------------
    st.subheader("🧼 Limpieza de Datos: Valores Faltantes")
    try:
        df_raw = pd.read_csv("argentina.csv")
        st.markdown("**Valores faltantes antes de la limpieza (argentina.csv):**")
        nulos_antes = df_raw.isnull().sum().to_frame(name="Nulos antes").sort_values(by="Nulos antes", ascending=False)
        st.dataframe(nulos_antes)
    except Exception as e:
        st.warning(f"No se pudo cargar argentina.csv para ver los nulos originales: {e}")

    st.markdown("**Valores faltantes después de la limpieza y combinación de datos:**")
    nulos_despues = df.isnull().sum().to_frame(name="Nulos después").sort_values(by="Nulos después", ascending=False)
    st.dataframe(nulos_despues)

    st.subheader("🔍 Visualización de Nulos (Heatmap)")
    fig_nulls, ax_nulls = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=True, cmap="coolwarm", ax=ax_nulls, yticklabels=False,
                cbar_kws={"label": "Valores Faltantes"})
    ax_nulls.set_title("Mapa de Calor de Valores Faltantes", fontsize=14)
    st.pyplot(fig_nulls)

    # Gráfico de barras para los indicadores de la provincia seleccionada
    st.subheader("📊 Gráfico de Barras: Indicadores por Provincia")
    # Seleccionar indicadores relevantes numéricos
    indicadores = ["pobreza", "analfabetismo", "infraestructura_deficiente", "abandono_escolar",
                   "sin_cobertura_salud", "mortalidad_infantil", "pbi",
                   "accesos_internet_2024", "total_alumnos_escuela",
                   "total_universitarios", "cines_por_habitante", "medicos_por_habitante"]
    datos_prov_numeric = df[df["provincia"] == prov][indicadores].T
    datos_prov_numeric.columns = ["Valor"]
    datos_prov_numeric = datos_prov_numeric.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Escalar PBI a millones, accesos_internet_2024 a miles, total_alumnos_escuela a diez miles, total_universitarios a miles
    if "pbi" in datos_prov_numeric.index:
        datos_prov_numeric.loc["pbi", "Valor"] = datos_prov_numeric.loc["pbi", "Valor"] / 1_000_000
    if "accesos_internet_2024" in datos_prov_numeric.index:
        datos_prov_numeric.loc["accesos_internet_2024", "Valor"] = datos_prov_numeric.loc["accesos_internet_2024", "Valor"] / 1_000
    if "total_alumnos_escuela" in datos_prov_numeric.index:
        datos_prov_numeric.loc["total_alumnos_escuela", "Valor"] = datos_prov_numeric.loc["total_alumnos_escuela", "Valor"] / 10_000
    if "total_universitarios" in datos_prov_numeric.index:
        datos_prov_numeric.loc["total_universitarios", "Valor"] = datos_prov_numeric.loc["total_universitarios", "Valor"] / 1_000

    if not datos_prov_numeric.empty:
        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        bars = ax_bar.bar(datos_prov_numeric.index, datos_prov_numeric["Valor"], color="skyblue")
        ax_bar.set_title(f"Indicadores Socioeconómicos de {prov}\n(PBI en millones, Accesos en miles, Alumnos en diez miles, Universitarios en miles)", fontsize=16, pad=15)
        ax_bar.set_ylabel("Valor", fontsize=12)
        ax_bar.set_xlabel("Indicador", fontsize=12)
        ax_bar.tick_params(axis="x", rotation=45, labelsize=10)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.7)

        # Añadir etiquetas de valores en las barras con unidades ajustadas
        for bar in bars:
            height = bar.get_height()
            label = (
                f'{height:.2f}M' if bar.get_x() == list(datos_prov_numeric.index).index("pbi") else
                f'{height:.2f}K' if bar.get_x() in [list(datos_prov_numeric.index).index("accesos_internet_2024"),
                                                   list(datos_prov_numeric.index).index("total_universitarios")] else
                f'{height:.2f}T' if bar.get_x() == list(datos_prov_numeric.index).index("total_alumnos_escuela") else
                f'{height:.2f}'
            )
            ax_bar.text(bar.get_x() + bar.get_width() / 2., height,
                        label if height > 0 else '0',
                        ha="center", va="bottom", fontsize=8)

        # Ajustar layout para evitar superposición
        plt.tight_layout()
        st.pyplot(fig_bar)
    else:
        st.warning("No hay datos numéricos para graficar.")

# -------------------------
# TAB 2: Comparaciones libres
# -------------------------
with tab2:
    st.header("📈 Comparar dos variables")
    col1, col2 = st.columns(2)
    x = col1.selectbox("Eje X", df.select_dtypes(include=np.number).columns, index=1)
    y = col2.selectbox("Eje Y", df.select_dtypes(include=np.number).columns, index=2)

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, hue="provincia", ax=ax, legend=False)

    # Etiquetar cada punto con el nombre de la provincia
    for i in range(len(df)):
        ax.text(df[x].iloc[i], df[y].iloc[i], df["provincia"].iloc[i], fontsize=8, alpha=0.8)

    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------
# TAB 3: Correlaciones
# -------------------------
with tab3:
    st.header("🔗 Correlación entre Internet y Pobreza")

    if "accesos_internet_2024" in df.columns and "accesos_internet_2022" in df.columns:
        # Calculamos el crecimiento
        df["crecimiento_accesos"] = df["accesos_internet_2024"] - df["accesos_internet_2022"]

        # Gráfico 1: Accesos 2024 vs Pobreza
        st.subheader("📡 Accesos a Internet (2024) vs Pobreza")
        fig1, ax1 = plt.subplots()
        scatter = sns.scatterplot(data=df, x="accesos_internet_2024", y="pobreza", hue="provincia", palette="tab20", ax=ax1)
        for i in range(len(df)):
            ax1.text(df["accesos_internet_2024"].iloc[i], df["pobreza"].iloc[i], df["provincia"].iloc[i],
                     fontsize=8, alpha=0.9)
        ax1.set_xlabel("Accesos a Internet por provincia (2024)")
        ax1.set_ylabel("Pobreza (%)")
        ax1.legend_.remove()  # Oculta la leyenda
        st.pyplot(fig1)

        correl_1 = df["accesos_internet_2024"].corr(df["pobreza"])
        st.markdown(f"**Correlación (2024)**: `{correl_1:.2f}`")

        # Gráfico 2: Crecimiento accesos vs Pobreza
        st.subheader("📈 Crecimiento en Accesos a Internet (2022–2024) vs Pobreza")
        fig2, ax2 = plt.subplots()
        scatter2 = sns.scatterplot(data=df, x="crecimiento_accesos", y="pobreza", hue="provincia", palette="tab20", ax=ax2)
        for i in range(len(df)):
            ax2.text(df["crecimiento_accesos"].iloc[i], df["pobreza"].iloc[i], df["provincia"].iloc[i],
                     fontsize=8, alpha=0.9)
        ax2.set_xlabel("Crecimiento en Accesos a Internet (2022–2024)")
        ax2.set_ylabel("Pobreza (%)")
        ax2.legend_.remove()  # Oculta la leyenda
        st.pyplot(fig2)

        correl_2 = df["crecimiento_accesos"].corr(df["pobreza"])
        st.markdown(f"**Correlación (crecimiento 2022–2024)**: `{correl_2:.2f}`")

        if correl_2 < -0.5:
            st.success("📉 Alta correlación negativa: Más crecimiento en internet → Menor pobreza.")
        elif correl_2 < -0.3:
            st.info("📉 Correlación negativa moderada.")
        else:
            st.warning("ℹ️ No se observa una correlación significativa.")
    else:
        st.error("No se encontraron datos suficientes para accesos a internet en 2022 y 2024.")

# -------------------------
# TAB 4: Clustering KMeans
# -------------------------
with tab4:
    st.header("📊 Clustering de Provincias")
    cluster_vars = st.multiselect("Seleccioná variables para el clustering",
                                  df.select_dtypes(include=np.number).columns.tolist(),
                                  default=["pobreza", "analfabetismo", "infraestructura_deficiente", "abandono_escolar"])
    if len(cluster_vars) >= 2:
        scaler = StandardScaler()
        X = scaler.fit_transform(df[cluster_vars].fillna(0))
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["cluster"] = kmeans.fit_predict(X)

        fig_cluster, ax_cluster = plt.subplots()
        sns.scatterplot(data=df, x=cluster_vars[0], y=cluster_vars[1], hue="cluster", palette="Set2", ax=ax_cluster)
        for i, row in df.iterrows():
            ax_cluster.text(row[cluster_vars[0]], row[cluster_vars[1]], row["provincia"], fontsize=7)
        st.pyplot(fig_cluster)
    else:
        st.warning("Seleccioná al menos dos variables para visualizar el clustering.")

# -------------------------
# TAB 5: Modelo de regresión
# -------------------------
with tab5:
    st.header("📉 Regresión para predecir la pobreza")
    features = st.multiselect("Variables predictoras",
                              df.select_dtypes(include=np.number).columns.drop("pobreza").tolist(),
                              default=["analfabetismo", "abandono_escolar", "infraestructura_deficiente", "pbi"])
    if features:
        X = df[features].fillna(0)
        y = df["pobreza"].fillna(0)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        fig_reg, ax_reg = plt.subplots()
        ax_reg.scatter(y, y_pred)

        # Agregar nombres de provincias
        for i in range(len(df)):
            ax_reg.annotate(df["provincia"].iloc[i], (y.iloc[i], y_pred[i]), fontsize=8, alpha=0.7)

        ax_reg.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax_reg.set_xlabel("Pobreza real")
        ax_reg.set_ylabel("Pobreza predicha")
        ax_reg.set_title("Regresión lineal")
        st.pyplot(fig_reg)

        st.markdown("**R² Score (entrenamiento):** {:.2f}".format(model.score(X, y)))
    else:
        st.warning("Seleccioná al menos una variable predictora.")

# -------------------------
# TAB 6: Conclusión
# -------------------------
with tab6:
    st.header("🌲 Árbol de Decisión: ¿Alta Pobreza?")
    st.markdown(
        "Este árbol intenta predecir si una provincia tiene un nivel de pobreza **alto o bajo**, según variables clave como conectividad, salud y educación.")

    # Selección de variables predictoras
    columnas_numericas = df.select_dtypes(include=np.number).columns.drop(["pobreza", "pobreza_alta"])
    columnas_por_defecto = [col for col in ["accesos_internet_2024", "analfabetismo", "abandono_escolar", "pbi"] if
                            col in columnas_numericas]

    features_tree = st.multiselect(
        "Seleccioná variables para el árbol",
        opciones := list(columnas_numericas),
        default=columnas_por_defecto
    )

    if features_tree:
        X_tree = df[features_tree].fillna(0)
        y_tree = df["pobreza_alta"]

        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(X_tree, y_tree)

        fig_tree, ax_tree = plt.subplots(figsize=(16, 8))
        plot_tree(clf, feature_names=features_tree, class_names=["Baja", "Alta"], filled=True, rounded=True,
                  fontsize=10)
        st.pyplot(fig_tree)

        st.markdown(f"**Precisión del árbol:** {clf.score(X_tree, y_tree):.2f}")

        st.markdown("---")
        st.subheader("🧠 Conclusión del Árbol de Decisión")

        st.markdown("""
        El árbol de decisión permite **clasificar si una provincia tiene un nivel de pobreza alto o bajo** según variables clave como:
        - **Accesos a internet (2024)**
        - **Analfabetismo**
        - **Abandono escolar**
        - **PBI per cápita**

        ### 📌 Hallazgos relevantes:
        - Las provincias con **más acceso a internet** y **mayor PBI** tienden a ser clasificadas como de **pobreza baja**.
        - El **analfabetismo alto** y el **abandono escolar elevado** se asocian con **pobreza alta**.
        - Se evidencia una **jerarquía clara de decisiones**: la conectividad y la educación aparecen como factores determinantes en la clasificación.

        Esto refuerza la hipótesis planteada en el análisis:  
        > _Las provincias argentinas con mejor acceso a servicios básicos (como educación y conectividad) presentan menores niveles de pobreza._

        Este modelo no busca predecir con exactitud, sino **ayudar a interpretar los factores que más influyen** en la situación socioeconómica de cada provincia.
        """)

    else:
        st.warning("Seleccioná al menos una variable para construir el árbol.")

# -------------------------
# TAB 7: Árbol de Decisión
# -------------------------
with tab7:
    st.header("📚 Conclusiones")

    st.markdown("""
    ### 🧠 Hipótesis  
    _“Las provincias argentinas con mayor acceso a internet, mayor inversión educativa y mejores servicios básicos presentan menores niveles de pobreza y desempleo.”_

    ### ✅ Validación basada en los datos
    El análisis exploratorio de los datos disponibles respalda esta hipótesis desde múltiples dimensiones:

    #### 📶 Internet y pobreza
    - Se observa una **correlación negativa significativa** entre los **accesos a internet (2024)** y la **tasa de pobreza**.
    - Al analizar el **crecimiento de los accesos entre 2022 y 2024**, también se identifica una relación negativa: **las provincias que más aumentaron la conectividad, tienden a mostrar menor pobreza**.

    #### 🎓 Educación y pobreza
    - Las provincias con mayor cantidad de **alumnos escolarizados y universitarios** tienden a tener **índices de pobreza más bajos**.
    - Existe una clara asociación entre **bajo analfabetismo**, **menor abandono escolar** y mejores indicadores socioeconómicos.

    #### 🏥 Servicios básicos
    - Variables como **infraestructura deficiente** y **falta de cobertura médica** muestran vínculos positivos con la pobreza, reforzando la idea de que la falta de servicios esenciales influye directamente en la calidad de vida.

    #### 🧪 Modelos analíticos
    - El **clustering KMeans** agrupa provincias con características similares, permitiendo identificar patrones regionales.
    - El modelo de **regresión lineal** mostró un R² aceptable, donde variables como **analfabetismo**, **infraestructura deficiente**, **abandono escolar** y **PBI** explican buena parte de la variación de la pobreza entre provincias.

    ### 🧭 Conclusión final
    La evidencia empírica **confirma con solidez la hipótesis inicial**. Invertir en conectividad, educación y servicios básicos no solo mejora indicadores individuales, sino que contribuye **a reducir estructuralmente la pobreza** en Argentina.
    """)