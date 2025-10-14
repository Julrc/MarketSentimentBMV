from flask import Flask, render_template
import pandas as pd
import plotly.express as px

app = Flask(__name__)

@app.route('/')
def index():
    try:
        # Load the Sentiment Index data
        sentiment_data = pd.read_csv('data/sentiment_index.csv')
        sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])

        # Generate the sentiment plot
        fig = px.line(
            sentiment_data, 
            x='Date', 
            y='Sentiment_Index', 
            labels={'Date': 'Date', 'Sentiment_Index': 'Sentiment Index'},
            width=1040,
            height=500,
        )

        # Update the layout for dark theme
        fig.update_layout(
            paper_bgcolor='#1e293b',  # Dark background
            plot_bgcolor='#1e293b',   # Dark background
            font_color='#e2e8f0',     # Light text
            title_font_color='#e2e8f0',
            title_font_size=24,       # Larger title
            margin=dict(t=50, r=30, b=50, l=50),  # Adjusted margins
    xaxis=dict(
        gridcolor='#475569',    # Lighter grid color
        linecolor='#475569',    # Lighter axis color
        zerolinecolor='#475569',
        showgrid=True,
        gridwidth=1,
        linewidth=2,
        title_font=dict(size=14),
        tickfont=dict(size=12),
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1M",
                     step="month",
                     stepmode="backward"),
                dict(count=3,
                     label="3M",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6M",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="1Y",
                     step="year",
                     stepmode="backward"),
                dict(step="all",
                     label="All")
            ]),
            bgcolor='#1e293b',        # Background color of the buttons
            activecolor='#60a5fa',    # Color of the active button
            font=dict(color='#e2e8f0') # Font color
        ),
        type="date"
    ),
    
            yaxis=dict(
                gridcolor='#475569',    # Lighter grid color
                linecolor='#475569',    # Lighter axis color
                zerolinecolor='#475569',
                showgrid=True,
                gridwidth=1,
                linewidth=2,
                title_font=dict(size=14),
                tickfont=dict(size=12),
            ),
            showlegend=False,
        )

        # Update line color and style
        fig.update_traces(
            line_color='#60a5fa',    # Light blue line
            line_width=2,            # Thicker line
            hovertemplate='<b>Date</b>: %{x}<br>' +
                        '<b>Sentiment</b>: %{y:.2f}<extra></extra>'
        )

        # Convert the plot to HTML components
        graph_html = fig.to_html(
            full_html=False, 
            include_plotlyjs='cdn',
            config={'displayModeBar': True,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                    'displaylogo': False}
        )
    
        # Load the news data
        news_data = pd.read_csv('news_titles.csv')
        
        # Debug: Print column names
        print("Column names:", news_data.columns.tolist())
        
        # Convert Date column to datetime
        news_data['Date'] = pd.to_datetime(news_data['Date'])
        # Format the date as string
        news_data['Date'] = news_data['Date'].dt.strftime('%Y-%m-%d')
        # Extract the 'title' and 'date' columns
        news_titles = news_data[['Date', 'title']].dropna()
        # Sort the news by date descending
        news_titles = news_titles.sort_values(by='Date', ascending=False)
        # Convert to list of dictionaries
        news_list = news_titles.to_dict(orient='records')
        
        return render_template('index.html', graph_html=graph_html, news_list=news_list)
    except Exception as e:
        # Handle exceptions and provide feedback
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
