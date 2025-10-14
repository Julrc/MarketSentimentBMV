# src/app.py
from pathlib import Path
from flask import Flask, render_template
import pandas as pd
import plotly.express as px

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
app = Flask(__name__,
            template_folder=str(BASE / "templates"),
            static_folder=str(BASE / "static"))

@app.route("/")
def index():
    try:
        sentiment_csv = ROOT / "data" / "sentiment_index.csv"
        news_csv = ROOT / "data" / "news_titles.csv"

        sentiment_data = pd.read_csv(sentiment_csv)
        sentiment_data["Date"] = pd.to_datetime(sentiment_data["Date"])

        fig = px.line(
            sentiment_data,
            x="Date",
            y="Sentiment_Index",
            labels={"Date": "Date", "Sentiment_Index": "Sentiment Index"},
            width=1040,
            height=500,
        )
        fig.update_layout(
            paper_bgcolor="#1e293b",
            plot_bgcolor="#1e293b",
            font_color="#e2e8f0",
            title_font_color="#e2e8f0",
            title_font_size=24,
            margin=dict(t=50, r=30, b=50, l=50),
            xaxis=dict(
                gridcolor="#475569",
                linecolor="#475569",
                zerolinecolor="#475569",
                showgrid=True,
                gridwidth=1,
                linewidth=2,
                title_font=dict(size=14),
                tickfont=dict(size=12),
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor="#1e293b",
                    activecolor="#60a5fa",
                    font=dict(color="#e2e8f0"),
                ),
                type="date",
            ),
            yaxis=dict(
                gridcolor="#475569",
                linecolor="#475569",
                zerolinecolor="#475569",
                showgrid=True,
                gridwidth=1,
                linewidth=2,
                title_font=dict(size=14),
                tickfont=dict(size=12),
            ),
            showlegend=False,
        )
        fig.update_traces(
            line_color="#60a5fa",
            line_width=2,
            hovertemplate="<b>Date</b>: %{x}<br><b>Sentiment</b>: %{y:.2f}<extra></extra>",
        )
        graph_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                "displaylogo": False,
            },
        )

        news_data = pd.read_csv(news_csv)
        news_data["Date"] = pd.to_datetime(news_data["Date"]).dt.strftime("%Y-%m-%d")
        news_titles = news_data[["Date", "title"]].dropna().sort_values("Date", ascending=False)
        news_list = news_titles.to_dict(orient="records")

        return render_template("index.html", graph_html=graph_html, news_list=news_list)
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)

