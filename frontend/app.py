import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import timedelta
import os

st.set_page_config(layout="wide", page_title="Alpha Forecast Desk", page_icon="₿")

# --- STYLES CSS PERSONNALISÉS ---
st.markdown("""<style>
    .main-title { text-align: left; font-size: 1.2rem; font-weight: 700; color: #00CCFF; margin-bottom: 20px; }
    .header-container { display: flex; flex-direction: column; align-items: center; margin-top: 20px; margin-bottom: 40px; width: 100%; }
    .top-line { display: flex; justify-content: flex-start; width: 100%; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
    .last-price-label { color: #00CCFF; font-size: 2.4rem; font-weight: 800; margin: 0; }
    .last-price-value { color: white; font-size: 2.4rem; font-weight: 800; margin: 0; margin-left: 20px; }
    .section-subtitle { text-align: center; font-size: 3.5rem; font-weight: 900; color: #DA70D6; text-transform: uppercase; letter-spacing: 5px; margin: 0; }
</style>""", unsafe_allow_html=True)

# --- BARRE LATÉRALE ---
st.sidebar.markdown("### 📁 Chargez vos données ici")
file = st.sidebar.file_uploader("Fichier CSV", type="csv")

if file is not None:
    df_raw = pd.read_csv(file)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="s")
    df_h = df_raw.sort_values("datetime").set_index("datetime").resample("1H").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    
    st.sidebar.markdown("### 🎯 Marqueurs d'Importance")
    show_month = st.sidebar.toggle("🔴 Passage de Mois", value=False)
    show_weekend = st.sidebar.toggle("🟡 Vendredi/Samedi", value=False)

    min_d, max_d = df_h.index[0].to_pydatetime(), df_h.index[-1].to_pydatetime()
    tr = st.sidebar.slider("Période", min_d, max_d, (min_d, max_d), format="DD/MM HH:mm")
    
    horizons_disponibles = [1, 2, 4, 6, 8, 12, 24]
    hor_sel = sorted(st.sidebar.multiselect("Horizons", horizons_disponibles, default=[1]))
    
    df_calc = df_h.loc[tr[0]:tr[1]]

    if len(df_calc) >= 168:
        try:
            # Le modèle utilise 168h au passé pour prédire 24h au futur
            BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
            payload_df = df_calc.tail(400).copy()
            payload_df.index = payload_df.index.strftime('%Y-%m-%d %H:%M:%S')
            
            with st.spinner("Analyse neuronale..."):
                response = requests.post(f"{BACKEND_URL}/predict", json=payload_df.to_dict(orient="index"), timeout=60)
            
            if response.status_code == 200:
                res = response.json()
                last_date = pd.to_datetime(res["last_date"])
                last_p = res["last_close"]

                # --- 1. GRAPHIQUE INTERACTIF (Zoom/Pan activé) ---
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_h.index, y=df_h["close"], name="Prix Réel", line=dict(color="rgba(255,255,255,0.15)", width=1)))
                fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc["close"], name="Observation", line=dict(color="#00CCFF", width=2)))

                # Marqueurs de début de mois (Points rouges uniques)
                if show_month:
                    month_starts = df_calc[df_calc.index.is_month_start].between_time('00:00', '00:00')
                    fig.add_trace(go.Scatter(x=month_starts.index, y=month_starts["close"], mode='markers', 
                                           marker=dict(color='red', size=12, symbol='circle'), name="Début de Mois"))

                # Marqueurs Weekend
                if show_weekend:
                    for w in df_calc[df_calc.index.weekday == 4].index:
                        fig.add_vline(x=w, line_width=1, line_dash="dash", line_color="yellow", opacity=0.3)

                # Courbe de prédiction TFT avec labels
                if hor_sel:
                    pred_x = [df_calc.index[-1]] + [last_date + timedelta(hours=h) for h in hor_sel]
                    pred_y = [last_p] + [res["median"][h-1] for h in hor_sel]
                    pred_labels = [""] + [f"H+{h}" for h in hor_sel]
                    fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode="lines+markers+text", text=pred_labels,
                                           textposition="top center", textfont=dict(color="white", size=14),
                                           line=dict(color="#DA70D6", width=3, dash='dash'), marker=dict(color="#DA70D6", size=8)))

                # Configuration de l'interactivité (Pan et Zoom à la molette)
                fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=0, t=20, b=0),
                                  dragmode='pan', xaxis=dict(showgrid=False), 
                                  yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"))
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

                # --- 2. HEADER PRIX ---
                st.markdown(f"""<div class="header-container"><div class="top-line">
                            <span class="last-price-label">BTC Price at {last_date.strftime("%d/%m/%Y %H:%M")} :</span>
                            <span class="last-price-value">${last_p:,.2f}</span></div>
                            <h1 class="section-subtitle">STRATEGY DECISIONS</h1></div>""", unsafe_allow_html=True)

                # --- 3. JAUGES ET DECISIONS ---
                cols = st.columns(len(hor_sel))
                for i, h in enumerate(hor_sel):
                    p_h = res["median"][h-1]
                    perf_pct = ((p_h - last_p) / last_p) * 100
                    confiance = 65.0  
                    
                    if abs(perf_pct) <= 0.15:
                        decision, btn_bg, color_trend = "HOLD", "#444", "#FFA500"
                    elif perf_pct > 0.15:
                        decision, btn_bg, color_trend = ("BUY", "#238636", "#00FF88") if confiance > 40 else ("WAIT", "#444", "#00FF88")
                    else:
                        decision, btn_bg, color_trend = ("SELL", "#DA3633", "#FF4B4B") if confiance > 40 else ("WAIT", "#444", "#FF4B4B")

                    with cols[i]:
                        st.markdown(f"<div style='text-align:center;'><p style='font-size:3.2rem; font-weight:bold; color:#DA70D6; margin:0;'>H+{h}</p>"
                                    f"<h2 style='color:{color_trend}; font-size:2.6rem; margin:0;'>${p_h:,.0f}</h2>"
                                    f"<p style='font-size:1.8rem; color:white;'>{perf_pct/100:+.2%}</p></div>", unsafe_allow_html=True)
                        
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", 
                            value=confiance,
                            number={'suffix': "%", 'font': {'size': 26, 'color': 'white'}, 'valueformat': ".1f"},
                            gauge={'axis': {'range': [0, 100], 'visible': False}, 
                                   'bar': {'color': color_trend}, 
                                   'bgcolor': "#222"}
                        ))
                        fig_gauge.update_layout(height=150, margin=dict(l=10, r=10, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_gauge, use_container_width=True, key=f"g_{h}", config={'displayModeBar': False})
                        
                        st.markdown(f"<div style='background:{btn_bg}; color:white; text-align:center; padding:15px; border-radius:8px; font-weight:bold; font-size:1.5rem;'>{decision}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur : {e}")
else:
    st.info("👈 Veuillez charger un fichier CSV pour activer le Trading Desk.")