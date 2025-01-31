import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.datasets import make_blobs

# ---- Caching population creation for performance ----
@st.cache_data
def generate_population(n_samples:int, cluster_std:float=1.0):
    """Generate and cache population scatter points with 2D features."""
    if n_samples < 2:
        n_samples = 2  # enforce at least 2 to avoid dimension issues
    x_vals, _ = make_blobs(n_samples=n_samples, n_features=2, centers=1, cluster_std=cluster_std, random_state=42)
    # x_vals is shape (n_samples, 2)
    return pd.DataFrame({'x': x_vals[:, 0], 'y': x_vals[:, 1]})

# ---- Epidemic Models ----
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def seir_model(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def sidr_model(y, t, N, beta, gamma, theta):
    S, I, D, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - (gamma + theta) * I
    dDdt = theta * I
    dRdt = gamma * I
    return dSdt, dIdt, dDdt, dRdt

def seird_model(y, t, N, beta, sigma, gamma, theta):
    S, E, I, D, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + theta) * I
    dDdt = theta * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dDdt, dRdt

def run_epidemic_simulation():
    """Runs the selected epidemic model with user-selected parameters."""
    st.title("Epidemic Simulation")

    # ---- USER INPUTS ----
    st.subheader("Simulation Parameters")
    model_choice = st.selectbox("Select Model:", ["SIR", "SEIR", "SIDR", "SEIRD"])

    # Basic population & initial conditions
    N = st.number_input("Total Population (N)", min_value=2, max_value=200000, value=1000, step=10)
    I0 = st.number_input("Initial Infected (I0)", min_value=0, max_value=200000, value=1)
    R0 = st.number_input("Initial Recovered (R0)", min_value=0, max_value=200000, value=0)
    D0 = st.number_input("Initial Deceased (D0)", min_value=0, max_value=200000, value=0)
    E0 = st.number_input("Initial Exposed (E0, for SEIR/SEIRD)", min_value=0, max_value=200000, value=0)

    # Ensure initial sums don't exceed N
    sum_init = I0 + R0 + D0 + E0
    if sum_init > N:
        st.warning("Initial states exceed total population. Clamping E0 to fit.")
        E0 = max(0, N - (I0 + R0 + D0))

    S0 = N - I0 - R0 - D0 - E0

    st.markdown("---")
    st.subheader("Disease & Timeline Parameters")
    beta = st.slider("Infection Rate (beta)", min_value=0.0, max_value=2.0, value=0.6, step=0.1)
    gamma = st.slider("Recovery Rate (gamma)", min_value=0.0, max_value=1.0, value=1.0/7.0, step=0.01)
    sigma = st.slider("Incubation Rate (sigma, SEIR)", min_value=0.0, max_value=1.0, value=1.0/5.0, step=0.01)
    theta = st.slider("Mortality Rate (theta, SIDR/SEIRD)", min_value=0.0, max_value=0.5, value=0.02, step=0.01)

    days = st.slider("Simulation Days", min_value=10, max_value=365, value=90, step=10)
    t = np.linspace(0, days-1, days)  # e.g., days=90 -> t=0..89

    # ---- Solve the chosen model ----
    if model_choice == "SIR":
        y0 = (S0, I0, R0)
        result = odeint(sir_model, y0, t, args=(N, beta, gamma))
        S, I, R = result.T
        E = np.zeros_like(S)
        D = np.zeros_like(S)
    elif model_choice == "SEIR":
        y0 = (S0, E0, I0, R0)
        result = odeint(seir_model, y0, t, args=(N, beta, sigma, gamma))
        S, E, I, R = result.T
        D = np.zeros_like(S)
    elif model_choice == "SIDR":
        y0 = (S0, I0, D0, R0)
        result = odeint(sidr_model, y0, t, args=(N, beta, gamma, theta))
        S, I, D, R = result.T
        E = np.zeros_like(S)
    elif model_choice == "SEIRD":
        y0 = (S0, E0, I0, D0, R0)
        result = odeint(seird_model, y0, t, args=(N, beta, sigma, gamma, theta))
        S, E, I, D, R = result.T

    # let user pick any day in range
    selected_day = st.slider("Select Day Index", 0, days-1, 0, step=1)
    selected_day = int(selected_day)  # ensure integer

    # ============ Plot: Epidemic Model ============
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, name="Susceptible", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=t, y=I, name="Infected", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=t, y=R, name="Recovered", line=dict(color="green")))
    if model_choice in ("SEIR", "SEIRD"):
        fig.add_trace(go.Scatter(x=t, y=E, name="Exposed", line=dict(color="orange")))
    if model_choice in ("SIDR", "SEIRD"):
        fig.add_trace(go.Scatter(x=t, y=D, name="Deceased", line=dict(color="black")))

    max_y = max(S.max(), I.max(), R.max(), E.max(), D.max())
    day_x = t[selected_day] if selected_day < len(t) else len(t)-1
    fig.add_trace(go.Scatter(
        x=[day_x, day_x],
        y=[0, max_y],
        mode="lines",
        name=f"Day {day_x}",
        line=dict(color="black", dash="dash")
    ))
    fig.update_layout(
        title=f"{model_choice} Model Simulation",
        xaxis_title="Days",
        yaxis_title="Population"
    )
    st.plotly_chart(fig)

    # ============ Plot: Scatter Population ============
    st.subheader("Population Scatter")
    pop_df = generate_population(N, cluster_std=1.0)  # Get scatter data from cache

    # Safeguard day index
    # If user picks day=days-1, ensure S, E, I, R, D exist
    s_val = S[selected_day] if selected_day < len(S) else S[-1]
    e_val = E[selected_day] if selected_day < len(E) else E[-1]
    i_val = I[selected_day] if selected_day < len(I) else I[-1]
    r_val = R[selected_day] if selected_day < len(R) else R[-1]
    d_val = D[selected_day] if selected_day < len(D) else D[-1]

    s_count = max(0, int(s_val))
    e_count = max(0, int(e_val))
    i_count = max(0, int(i_val))
    r_count = max(0, int(r_val))
    d_count = max(0, int(d_val))

    total_comp = s_count + e_count + i_count + r_count + d_count
    if total_comp > N:
        # reduce s_count if compartments sum up beyond N
        s_count = max(0, N - (e_count + i_count + r_count + d_count))

    # Use a Python list of strings, not a NumPy array
    colors = ["blue"] * N
    idx = 0
    # Exposed
    if e_count > 0:
        for i in range(e_count):
            if idx+i < N:
                colors[idx + i] = "orange"
        idx += e_count
    # Infected
    if i_count > 0:
        for i in range(i_count):
            if idx+i < N:
                colors[idx + i] = "red"
        idx += i_count
    # Recovered
    if r_count > 0:
        for i in range(r_count):
            if idx+i < N:
                colors[idx + i] = "green"
        idx += r_count
    # Deceased
    if d_count > 0:
        for i in range(d_count):
            if idx+i < N:
                colors[idx + i] = "black"
        idx += d_count
    # remainder: "blue"

    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=pop_df["x"],
        y=pop_df["y"],
        mode="markers",
        marker=dict(color=colors, size=6, opacity=0.8)
    ))
    scatter_fig.update_layout(
        xaxis_showgrid=False, yaxis_showgrid=False,
        xaxis_title=None, yaxis_title=None,
        xaxis_ticks='', yaxis_ticks='',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        title=f"Population Scatter on Day {day_x}"
    )
    st.plotly_chart(scatter_fig)

    # ============ (Optional) Pie Chart ============
    st.subheader("Pie Chart of Population Breakdown")

    labels = ["Susceptible", "Infected", "Recovered"]
    values = [s_val, i_val, r_val]
    color_seq = ["blue", "red", "green"]

    if model_choice in ("SEIR", "SEIRD"):
        labels.append("Exposed")
        values.append(e_val)
        color_seq.append("orange")
    if model_choice in ("SIDR", "SEIRD"):
        labels.append("Deceased")
        values.append(d_val)
        color_seq.append("black")

    pie_fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(colors=color_seq)
    )])
    pie_fig.update_layout(title_text=f"Population Breakdown on Day {day_x}")
    st.plotly_chart(pie_fig)

if __name__ == "__main__":
    run_epidemic_simulation()
