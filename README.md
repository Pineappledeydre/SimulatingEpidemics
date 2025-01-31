# SimulatingEpidemics

This Streamlit application models the spread of infectious diseases using various compartmental models, including SIR, SEIR, SIDR, and SEIRD. Users can adjust parameters to observe how different factors influence disease dynamics over time.

## Features

- **Interactive Parameter Adjustment**: Modify parameters such as infection rate, recovery rate, incubation rate, and mortality rate to see their effects on the simulation.
- **Multiple Models**: Choose from SIR, SEIR, SIDR, and SEIRD models to simulate different disease progression scenarios.
- **Dynamic Visualizations**: View time-series plots of susceptible, infected, recovered, exposed, and deceased populations. Additionally, observe scatter plots representing the population distribution and pie charts illustrating population breakdowns.

## Installation

To run this application locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pineappledeydre/SimulatingEpidemics.git
   cd SimulatingEpidemics
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7 or later installed. Install the required Python packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the Streamlit app with:
   ```bash
   streamlit run streamlit_app.py
   ```

   The application will open in your default web browser.

## Usage

Upon launching the app:

1. **Select the Model**: Choose between SIR, SEIR, SIDR, or SEIRD models.
2. **Set Population Parameters**: Define the total population and initial counts for susceptible, infected, recovered, deceased, and exposed individuals.
3. **Adjust Disease Parameters**: Use sliders to set values for infection rate (`beta`), recovery rate (`gamma`), incubation rate (`sigma`), and mortality rate (`theta`).
4. **Simulation Duration**: Select the number of days to simulate.
5. **View Results**: The app will display:
   - A line chart showing the progression of each compartment over time.
   - A scatter plot representing the population distribution on a selected day.
   - A pie chart illustrating the population breakdown on a selected day.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.
