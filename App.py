
# STREAMLIT DEPLOYMENT FOR SVM MODEL
# ======================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Water Network SVM Monitor",
    page_icon="💧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.header {
    font-size: 28px !important;
    color: #1f77b4;
    padding-bottom: 10px;
    border-bottom: 2px solid #f0f2f6;
    margin-bottom: 20px;
}
.prediction-card {
    border-radius: 10px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.leak {
    background-color: #fff6f6;
    border-left: 6px solid #ff4444;
}
.no-leak {
    background-color: #f6fff6;
    border-left: 6px solid #44cc44;
}
.sidebar-section {
    margin-bottom: 25px;
}
.metric-box {
    background-color: #f9f9f9;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Load the saved SVM model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_water_model.pkl")
        st.success("SVM model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading SVM model: {str(e)}")
        return None

# Feature engineering class (must match training)
class FeatureEngineer:
    def __init__(self):
        self.hourly_demand_means_ = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.hourly_demand_means_ = X.groupby('Hour_of_Day')['Consumer_Demand_Liters'].mean()
        return self
        
    def transform(self, X):
        X = X.copy()
        X['Pressure_Demand_Ratio'] = X['Pressure_PSI'] / (X['Consumer_Demand_Liters'] + 1e-6)
        
        if self.hourly_demand_means_ is not None:
            X['Hourly_Demand_Deviation'] = X['Consumer_Demand_Liters'] - \
                                          X['Hour_of_Day'].map(self.hourly_demand_means_)
        else:
            # Fallback if not fitted (shouldn't happen with saved pipeline)
            X['Hourly_Demand_Deviation'] = 0
            
        return X

# Main app function
def main():
    st.markdown('<div class="header">💧 Water Network Optimization - SVM Model</div>', 
                unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Sidebar controls
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Network Parameters")
        
        current_time = datetime.now()
        current_hour = current_time.hour
        
        col1, col2 = st.columns(2)
        with col1:
            flow_rate = st.slider("Flow Rate (LPS)", 50, 500, 200, key='flow')
            pressure = st.slider("Pressure (PSI)", 20, 100, 60, key='pressure')
        with col2:
            demand = st.slider("Demand (Liters)", 100, 300, 180, key='demand')
            hour_of_day = st.selectbox("Hour of Day", list(range(24)), current_hour, key='hour')
        
        district = st.selectbox("District", ["North", "South", "East", "West"], key='district')
        pump_status = st.radio("Pump Status", ["ON", "OFF"], key='pump')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Model Type", "Support Vector Machine (SVM)")
        st.metric("Test Accuracy", "92.3%")  # Replace with your actual accuracy
        st.metric("Optimal Pressure Range", "40-80 PSI")
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Main content
    tab1, tab2 = st.tabs(["Real-time Monitoring", "Network Insights"])
    
    with tab1:
        # Prepare input data
        input_data = {
            'Flow_Rate_LPS': flow_rate,
            'Pressure_PSI': pressure,
            'Consumer_Demand_Liters': demand,
            'Temperature_C': 25.0,  # Default value
            'Rainfall_mm': 0.0,     # Default value
            'Hour_of_Day': hour_of_day,
            'District': district,
            'Pump_Status': pump_status
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            # Display results
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.subheader("Leak Detection Status")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-card leak">
                        <h2>🚨 LEAK DETECTED</h2>
                        <div class="metric-box">
                            <h3>Confidence: {proba:.1%}</h3>
                            <p><b>Risk Level:</b> High</p>
                        </div>
                        <h4>Recommended Actions:</h4>
                        <ol>
                            <li>Dispatch team to {district} district</li>
                            <li>Reduce pressure to 40-50 PSI</li>
                            <li>Check nearest junctions</li>
                        </ol>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card no-leak">
                        <h2>✅ SYSTEM NORMAL</h2>
                        <div class="metric-box">
                            <h3>Confidence: {(1-proba):.1%}</h3>
                            <p><b>Risk Level:</b> Low</p>
                        </div>
                        <h4>Maintenance Tips:</h4>
                        <ul>
                            <li>Monitor pressure fluctuations</li>
                            <li>Check meter calibrations weekly</li>
                            <li>Inspect pumps monthly</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Network Health Visualization")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create decision boundary visualization (simplified for SVM)
                xx, yy = np.meshgrid(
                    np.linspace(100, 300, 50),  # Demand range
                    np.linspace(20, 100, 50)    # Pressure range
                )
                
                # Create test points for visualization
                test_points = pd.DataFrame({
                    'Flow_Rate_LPS': flow_rate,
                    'Pressure_PSI': yy.ravel(),
                    'Consumer_Demand_Liters': xx.ravel(),
                    'Temperature_C': 25.0,
                    'Rainfall_mm': 0.0,
                    'Hour_of_Day': hour_of_day,
                    'District': district,
                    'Pump_Status': pump_status
                })
                
                # Add engineered features
                fe = FeatureEngineer()
                fe.hourly_demand_means_ = model.named_steps['features'].hourly_demand_means_
                test_points = fe.transform(test_points)
                
                # Predict and plot
                Z = model.predict_proba(test_points)[:, 1].reshape(xx.shape)
                contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.6)
                plt.colorbar(contour, label='Leak Probability')
                
                # Plot current state
                ax.scatter(demand, pressure, color='purple', s=200, 
                          label=f'Current State ({district})')
                
                # Threshold lines
                ax.axhline(40, color='red', linestyle='--', label='Pressure Threshold (40 PSI)')
                ax.axvline(200, color='orange', linestyle='--', label='Demand Threshold (200 L)')
                
                ax.set_title("SVM Decision Boundaries (Pressure vs Demand)")
                ax.set_xlabel("Water Demand (Liters)")
                ax.set_ylabel("Pressure (PSI)")
                ax.legend()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    with tab2:
        st.subheader("Network Performance Analytics")
        
        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Avg Pressure", f"{data['Pressure_PSI'].mean():.1f} PSI")
            st.metric("Pressure Std Dev", f"{data['Pressure_PSI'].std():.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with m2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Avg Demand", f"{data['Consumer_Demand_Liters'].mean():.1f} L")
            st.metric("Peak Demand", f"{data['Consumer_Demand_Liters'].max():.1f} L")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with m3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Leak Frequency", f"{data['Leak_Detected'].mean():.2%}")
            st.metric("Most Vulnerable District", 
                     data.groupby('District')['Leak_Detected'].mean().idxmax())
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Charts
        st.subheader("Historical Patterns")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
        # District analysis
        district_stats = data.groupby('District').agg({
            'Leak_Detected': 'mean',
            'Pressure_PSI': 'median',
            'Consumer_Demand_Liters': 'median'
        }).sort_values('Leak_Detected', ascending=False)
        
        district_stats['Leak_Detected'].plot(kind='bar', ax=ax1, color='salmon')
        ax1.set_title("Leak Frequency by District")
        ax1.set_ylabel("Probability of Leak")
        ax1.axhline(0.1, color='red', linestyle='--', alpha=0.5)
        
        # Highlight current district
        if district in district_stats.index:
            idx = list(district_stats.index).index(district)
            ax1.patches[idx].set_edgecolor('black')
            ax1.patches[idx].set_linewidth(2)
        
        # Hourly patterns
        hourly_stats = data.groupby('Hour_of_Day').agg({
            'Consumer_Demand_Liters': 'median',
            'Leak_Detected': 'mean'
        })
        
        hourly_stats['Consumer_Demand_Liters'].plot(
            ax=ax2, color='royalblue', label='Demand'
        )
        ax2.set_title("Daily Demand and Leak Patterns")
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Median Demand (Liters)")
        
        ax2b = ax2.twinx()
        hourly_stats['Leak_Detected'].plot(
            ax=ax2b, color='red', linestyle='--', label='Leak Probability'
        )
        ax2b.set_ylabel("Leak Probability")
        
        # Combine legends
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        # Mark current hour
        ax2.axvline(hour_of_day, color='green', linestyle=':', alpha=0.7)
        
        st.pyplot(fig)

if __name__ == "__main__":
    # Load your data (replace with your actual data loading)
    data = pd.read_csv("optimized_water_supply_data.csv")
    main()