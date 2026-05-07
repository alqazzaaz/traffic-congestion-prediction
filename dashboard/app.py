import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Early Warning System",
    page_icon="🚦",
    layout="wide"
)

# ─────────────────────────────────────────
# Load model and data
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/xgboost_best.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

@st.cache_data
def load_test_data():
    X = pd.read_csv('data/processed/X_test.csv')
    y = pd.read_csv('data/processed/y_test.csv').squeeze()
    return X, y

model, scaler = load_model()
X_test, y_test = load_test_data()

# ─────────────────────────────────────────
# Warning thresholds
# ─────────────────────────────────────────
THRESHOLD_GREEN  = 0.40
THRESHOLD_RED    = 0.65

def get_warning_level(prob):
    if prob < THRESHOLD_GREEN:
        return "GREEN", "●", "No congestion expected", "#2ecc71"
    elif prob < THRESHOLD_RED:
        return "YELLOW", "●", "Congestion possible – monitor", "#f39c12"
    else:
        return "RED", "●", "WARNING: Congestion likely in 20 min!", "#e74c3c"

def predict(features_df):
    prob = model.predict_proba(features_df)[0][1]
    return prob

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("🚦 Traffic Congestion Early Warning System")
st.markdown("**ML-based 20-minute congestion prediction for Smart City applications**")
st.markdown("*Bachelor Thesis – TU Dortmund | XGBoost Model | Guangzhou Urban Traffic Dataset*")
st.divider()

# ─────────────────────────────────────────
# Mode selection
# ─────────────────────────────────────────
mode = st.radio(
    "Select Mode:",
    ["🎮 Manual Simulation", "▶️ Replay Real Traffic Data"],
    horizontal=True
)

st.divider()

# ─────────────────────────────────────────
# MODE 1 – Manual Simulation
# ─────────────────────────────────────────
if mode == "🎮 Manual Simulation":

    st.sidebar.header("📍 Traffic Parameters")
    st.sidebar.markdown("Adjust parameters to simulate:")

    road_id = st.sidebar.slider("Road Segment ID", 1, 214, 107)
    hour = st.sidebar.slider("Hour of Day", 0, 23, 8)
    weekday = st.sidebar.selectbox(
        "Day of Week",
        options=[0,1,2,3,4,5,6],
        format_func=lambda x: ['Monday','Tuesday','Wednesday',
                                'Thursday','Friday',
                                'Saturday','Sunday'][x])
    current_speed = st.sidebar.slider("Current Speed (km/h)", 1, 100, 45)
    speed_10min = st.sidebar.slider("Speed 10 min ago (km/h)", 1, 100, 48)
    speed_20min = st.sidebar.slider("Speed 20 min ago (km/h)", 1, 100, 50)
    speed_30min = st.sidebar.slider("Speed 30 min ago (km/h)", 1, 100, 52)
    speed_60min = st.sidebar.slider("Speed 60 min ago (km/h)", 1, 100, 55)

    is_weekend = 1 if weekday >= 5 else 0
    is_rush_hour = 1 if (hour in [7,8,9,17,18,19]
                         and is_weekend == 0) else 0
    rolling_mean = np.mean([current_speed,
                            speed_10min, speed_20min])
    rolling_std = np.std([current_speed,
                          speed_10min, speed_20min])

    def normalize(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    features = {
        'road_id':              normalize(road_id, 1, 214),
        'hour':                 normalize(hour, 0, 23),
        'weekday':              normalize(weekday, 0, 6),
        'is_weekend':           float(is_weekend),
        'is_rush_hour':         float(is_rush_hour),
        'speed':                normalize(current_speed, 1, 100),
        'speed_lag_1':          normalize(speed_10min, 1, 100),
        'speed_lag_2':          normalize(speed_20min, 1, 100),
        'speed_lag_3':          normalize(speed_30min, 1, 100),
        'speed_lag_6':          normalize(speed_60min, 1, 100),
        'speed_rolling_mean_3': normalize(rolling_mean, 1, 100),
        'speed_rolling_std_3':  normalize(rolling_std, 0, 50)
    }

    prob = predict(pd.DataFrame([features]))
    level, emoji, message, color = get_warning_level(prob)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Current Status")
        st.markdown(f"""
<div style='background-color:{color}22;
            border-left: 5px solid {color};
            padding: 20px; border-radius: 8px;'>
    <h2 style='margin:0'>
        <span style='color:{color}; 
                     font-size:40px'>●</span>
        <span style='color:{color}'> {level}</span>
    </h2>
    <p style='font-size:16px; margin-top:10px'>
        {message}
    </p>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Congestion Probability")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40],
                     'color': "rgba(46,204,113,0.2)"},
                    {'range': [40, 65],
                     'color': "rgba(243,156,18,0.2)"},
                    {'range': [65, 100],
                     'color': "rgba(231,76,60,0.2)"}
                ],
                'threshold': {
                    'line': {'color': color, 'width': 4},
                    'thickness': 0.75,
                    'value': prob * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=250, margin=dict(t=30,b=0))
        st.plotly_chart(fig_gauge,
                        use_container_width=True)

    with col3:
        st.markdown("### Traffic Summary")
        st.metric("Current Speed", f"{current_speed} km/h",
                  delta=f"{current_speed-speed_10min} "
                        f"km/h vs 10min ago")
        st.metric("Rush Hour",
                  "Yes ⚠️" if is_rush_hour else "No ✅")
        st.metric("Weekend",
                  "Yes" if is_weekend else "No")

    st.divider()
    st.markdown("### 📈 Speed Trend (Last 60 Minutes)")

    speed_history = {
        '60 min ago': speed_60min,
        '30 min ago': speed_30min,
        '20 min ago': speed_20min,
        '10 min ago': speed_10min,
        'Now':        current_speed
    }

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=list(speed_history.keys()),
        y=list(speed_history.values()),
        mode='lines+markers',
        line=dict(color='steelblue', width=3),
        marker=dict(size=10),
        name='Speed'
    ))
    fig_trend.add_hline(
        y=30, line_dash="dash",
        line_color="red",
        annotation_text="Congestion threshold (30 km/h)")
    fig_trend.update_layout(
        yaxis_title="Speed (km/h)",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ─────────────────────────────────────────
# MODE 2 – Replay Real Traffic Data
# ─────────────────────────────────────────
else:
    st.markdown("### ▶️ Replay Real Traffic Data")
    st.markdown("Select a road segment and watch real "
                "traffic data replay with live predictions.")

    col_a, col_b = st.columns(2)
    with col_a:
        selected_road = st.selectbox(
            "Select Road Segment:",
            options=sorted(X_test['road_id'].unique()),
            format_func=lambda x: f"Road {int(x*213)+1}"
        )
    with col_b:
        replay_speed = st.select_slider(
            "Replay Speed:",
            options=["Slow", "Normal", "Fast"],
            value="Normal"
        )

    speed_map = {"Slow": 1.0, "Normal": 0.3, "Fast": 0.05}
    delay = speed_map[replay_speed]

    # Filter data for selected road
    road_data = X_test[
        X_test['road_id'] == selected_road
    ].reset_index(drop=True)
    road_labels = y_test[
        X_test['road_id'] == selected_road
    ].reset_index(drop=True)

    st.info(f"Road segment has "
            f"{len(road_data):,} time steps available")

    # Limit to 144 steps = 1 full day
    road_data = road_data.head(144)
    road_labels = road_labels.head(144)

    if st.button("▶️ Start Replay", type="primary"):

        # Placeholders for live updates
        status_placeholder = st.empty()
        gauge_placeholder = st.empty()
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()

        prob_history = []
        speed_history = []
        level_history = []
        step_labels = []

        for i, (idx, row) in enumerate(
                road_data.iterrows()):

            prob = predict(pd.DataFrame([row]))
            level, emoji, message, color = \
                get_warning_level(prob)

            # Denormalize speed for display
            speed_kmh = row['speed'] * 99 + 1

            prob_history.append(prob * 100)
            speed_history.append(speed_kmh)
            level_history.append(level)
            hour_of_day = (i * 10) // 60
            minute = (i * 10) % 60
            step_labels.append(f"{hour_of_day:02d}:{minute:02d}")

            # Status box
            with status_placeholder.container():
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div style='background:{color}22;
                         border-left:5px solid {color};
                         padding:15px;
                         border-radius:8px;'>
                        <h2 style='color:{color};margin:0'>
                            {emoji} {level}
                        </h2>
                        <p>{message}</p>
                        <p>Time: {hour_of_day:02d}:{minute:02d}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.metric(
                        "Congestion Probability",
                        f"{prob:.1%}")
                    st.metric(
                        "Current Speed",
                        f"{speed_kmh:.1f} km/h")
                with c3:
                    st.metric(
                        "Actual Congestion",
                        "Yes 🔴" if road_labels[i] == 1
                        else "No 🟢")
                    correct = (
                        (prob >= 0.5) == (road_labels[i] == 1))
                    st.metric(
                        "Prediction",
                        "✅ Correct" if correct
                        else "❌ Wrong")

            # Live chart
            if len(prob_history) > 1:
                fig_live = go.Figure()

                # Speed line
                fig_live.add_trace(go.Scatter(
                    x=step_labels,
                    y=speed_history,
                    name='Speed (km/h)',
                    line=dict(color='steelblue',
                              width=2),
                    yaxis='y1'
                ))

                # Probability line
                fig_live.add_trace(go.Scatter(
                    x=step_labels,
                    y=prob_history,
                    name='Congestion Prob (%)',
                    line=dict(color='red',
                              width=2,
                              dash='dot'),
                    yaxis='y2'
                ))

                fig_live.add_hline(
                    y=30,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="30 km/h threshold")

                fig_live.update_layout(
                    title="Live Traffic Replay",
                    yaxis=dict(
                        title='Speed (km/h)',
                        range=[0, 100]),
                    yaxis2=dict(
                        title='Probability (%)',
                        overlaying='y',
                        side='right',
                        range=[0, 100]),
                    height=350,
                    margin=dict(t=40, b=20),
                    legend=dict(
                        orientation='h',
                        y=1.1)
                )

                chart_placeholder.plotly_chart(
                    fig_live,
                    use_container_width=True)

            time.sleep(delay)

        st.success(f"✅ Replay complete! "
                   f"Processed {len(road_data)} "
                   f"time steps.")

        # Summary stats
        st.markdown("### Replay Summary")
        red_count = level_history.count("RED")
        yellow_count = level_history.count("YELLOW")
        green_count = level_history.count("GREEN")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🔴 Red Warnings", red_count)
        with c2:
            st.metric("🟡 Yellow Warnings", yellow_count)
        with c3:
            st.metric("🟢 Green Steps", green_count)

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
st.markdown("### ℹ️ System Information")
col4, col5, col6 = st.columns(3)
with col4:
    st.info("**Model:** XGBoost\n\n"
            "**AUC-ROC:** 0.9604\n\n"
            "**Recall:** 86%")
with col5:
    st.info("**Dataset:** Guangzhou Urban Traffic\n\n"
            "**Roads:** 214 segments\n\n"
            "**Interval:** 10 minutes")
with col6:
    st.info("**Prediction horizon:** 20 minutes\n\n"
            "**Red threshold:** 65%\n\n"
            "**Green threshold:** 40%")