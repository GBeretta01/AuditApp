import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.decomposition import PCA

# Configuración inicial de la página
st.set_page_config(page_title="Analizador de Fraude", layout="wide")

# Función principal de la aplicación
def main_app():
    # Acceso a datos cargados
    data = st.session_state.data
    variables = ['Total_Inventario', 'Total_CxC', 'Total_CxP', 'Total_Ingresos', 'Total_Gastos']
    
    # Sidebar: Controles
    st.sidebar.header("⚙️ Configuración")
    contamination = st.sidebar.slider("Nivel de Sensibilidad (% Anomalías)", 1, 20, 5) / 100
    selected_vars = st.sidebar.multiselect("Variables para Análisis", variables, default=variables[:2])

    # Procesamiento de datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[variables])
    data_scaled = pd.DataFrame(data_scaled, columns=variables)
    data = pd.concat([data, data_scaled.add_suffix('_Scaled')], axis=1)

    # Modelo de detección de anomalías
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(data_scaled)
    data['Score_Anomalia'] = model.decision_function(data_scaled)
    data['Probabilidad_Fraude'] = np.interp(data['Score_Anomalia'], 
                                          (data['Score_Anomalia'].min(), data['Score_Anomalia'].max()), 
                                          (100, 0))
    data['Riesgo'] = np.where(data['Score_Anomalia'] < np.percentile(data['Score_Anomalia'], 100*contamination), 
                            'Alto Riesgo', 'Bajo Riesgo')

    # Dividir en pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Resumen", "📈 Relaciones", "🕰 Tendencias", "🔍 Profundidad", "🧮 Métricas"])

    with tab1:
        st.header("Resumen de Riesgos")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Empresas Analizadas", len(data['Entidad'].unique()))
            st.metric("% Alto Riesgo", f"{(data['Riesgo'] == 'Alto Riesgo').mean()*100:.1f}%")
            
        with col2:
            st.metric("Años Analizados", f"{int(data['Año'].min())} - {int(data['Año'].max())}")
            st.metric("Variables Clave", ", ".join(variables))
        
        st.dataframe(
            data[['Entidad', 'Año', 'Riesgo', 'Probabilidad_Fraude'] + variables],
            height=500,
            column_config={
                "Probabilidad_Fraude": st.column_config.ProgressColumn(
                    format="%.0f%%",
                    min_value=0,
                    max_value=100
                )
            }
        )

    with tab2:
        st.header("Análisis de Relaciones")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Matriz de Correlación")
            corr_matrix = data[variables].corr()
            fig = px.imshow(corr_matrix, 
                           labels=dict(color="Correlación"),
                           color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Relaciones Multivariables")
            fig = px.scatter_matrix(data, dimensions=selected_vars,
                                   color="Riesgo", hover_name="Entidad")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Análisis Temporal")
        empresa = st.selectbox("Seleccionar Empresa:", data['Entidad'].unique())
        df_empresa = data[data['Entidad'] == empresa]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Evolución Absoluta")
            fig = px.line(df_empresa, x='Año', y=variables, markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Evolución Normalizada")
            fig = px.line(df_empresa, x='Año', y=[f'{v}_Scaled' for v in variables], markers=True)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Análisis Profundo")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución por Riesgo")
            fig = px.violin(data, x="Riesgo", y=selected_vars, box=True)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Visualización 3D")
            pca = PCA(n_components=3)
            components = pca.fit_transform(data_scaled)
            fig = px.scatter_3d(components, x=0, y=1, z=2, color=data['Riesgo'],
                               labels={'0':'PC1', '1':'PC2', '2':'PC3'})
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.header("Métricas Detalladas")
        empresa_sel = st.selectbox("Seleccionar Entidad:", ['Todas'] + list(data['Entidad'].unique()))
        
        if empresa_sel == 'Todas':
            df_metrics = data.groupby('Entidad').agg({
                'Total_Inventario': 'sum',
                'Total_CxC': 'sum',
                'Total_Ingresos': 'sum',
                'Total_Gastos': 'sum'
            }).reset_index()
        else:
            df_metrics = data[data['Entidad'] == empresa_sel]
            
        st.dataframe(
            df_metrics.style.format(precision=0, thousands=","),
            height=400,
            use_container_width=True
        )

# Página de carga de datos
if 'data_loaded' not in st.session_state:
    st.title("🔼 Cargar Datos Contables")
    st.write("Suba un archivo CSV con el formato requerido para comenzar el análisis")
    
    uploaded_file = st.file_uploader("Seleccionar archivo CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Validar y cargar datos
            data = pd.read_csv(uploaded_file)
            required_columns = ['Entidad', 'Año', 'Total_Inventario', 'Total_CxC',
                               'Total_CxP', 'Total_Ingresos', 'Total_Gastos']
            
            if all(col in data.columns for col in required_columns):
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.success("✅ Archivo cargado correctamente!")
                
                if st.button("🚀 Iniciar Análisis", type="primary"):
                    st.rerun()
            else:
                missing = [col for col in required_columns if col not in data.columns]
                st.error(f"❌ Faltan columnas requeridas: {', '.join(missing)}")
                
        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {str(e)}")
else:
    main_app()

st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Asegúrese que el CSV contenga las columnas requeridas: Entidad, Año, Total_Inventario, Total_CxC, Total_CxP, Total_Ingresos, Total_Gastos")