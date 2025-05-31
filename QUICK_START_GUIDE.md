# Quick Start Guide: Video Game Sales Analysis

This guide will help you get started with the Video Game Sales Analysis application in just a few minutes.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (optional, for cloning the repository)

### Setup Steps

1. **Get the code**

   Either clone the repository:
   ```bash
   git clone https://github.com/rezashariatmadar/game-sales-analysis.git
   cd game-sales-analysis
   ```
   
   Or download and extract the ZIP file from GitHub.

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   python -m streamlit run app.py
   ```

4. **Access the dashboard**

   Open your web browser and go to: http://localhost:8501

## Using the Dashboard

### Analysis Tab

![Analysis Tab](./plots/analysis_tab.png)

1. **Use the sidebar filters**:
   - Select year range using the slider
   - Choose genres from the dropdown
   - Select platforms from the dropdown
   - Filter by publishers

2. **Explore visualizations**:
   - Sales trends chart shows how sales have changed over time
   - Regional distribution shows sales across different markets
   - Platform comparison shows which platforms perform best
   - Genre analysis shows the most popular game types

3. **View key metrics** at the top of the page for quick insights.

### Prediction Tab

![Prediction Tab](./plots/prediction_tab.png)

1. **Enter game details**:
   - Critic score (0-100)
   - Release year
   - Genre
   - Publisher
   - Regional sales estimates (optional)

2. **Click "Predict Sales"** to get:
   - Estimated total sales
   - Sales classification (high/low)
   - Confidence score

3. **Use "Reset Form"** to start over with a new prediction.

### Reports Tab

![Reports Tab](./plots/reports_tab.png)

Browse through different analysis reports by clicking on the options in the sidebar.

## Troubleshooting

### Common Issues

1. **"Module not found" error**:
   ```
   pip install -r requirements.txt
   ```

2. **"Command not found: streamlit"**:
   ```
   python -m streamlit run app.py
   ```

3. **Slow performance**:
   - Try filtering to a smaller data range
   - Restart the application

4. **Visualization not showing**:
   - Check that your filters aren't excluding all data
   - Try resetting filters to default values

## Next Steps

For more detailed information, refer to:
- [Full Documentation](./DOCUMENTATION.md)
- [Developer Guide](./DEVELOPER_GUIDE.md) 