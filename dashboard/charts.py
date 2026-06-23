# Figure-building helper functions used by the dashboard.

import plotly.express as px
import plotly.graph_objects as go

from constants import DEFAULT_BG, DEFAULT_FONT_COLOR

# fits in ./dashboard/charts.py
def style_figure(fig, y_range=None, x_reverse=False):
    if y_range is not None:
        fig.update_layout(yaxis_range=y_range)
    if x_reverse:
        fig.update_layout(xaxis_autorange="reversed")
    fig.update_layout(font_color=DEFAULT_FONT_COLOR, plot_bgcolor=DEFAULT_BG, paper_bgcolor=DEFAULT_BG)
    return fig


# fits in ./dashboard/charts.py
def build_bar_chart(df, x, y, color, title, color_sequence, y_range=None):
    filtered_df = df.loc[df[y].notna() & (df[y] != 0)].sort_values(y, ascending=False)
    fig = px.bar(filtered_df, x=x, y=y, color=color, title=title, text_auto=".3f", color_discrete_sequence=color_sequence)
    return style_figure(fig, y_range=y_range)


# fits in ./dashboard/charts.py
def build_scatter_chart(df, x, y, color, text, title, color_sequence, x_reverse=False):
    fig = px.scatter(df, x=x, y=y, color=color, text=text, title=title, color_discrete_sequence=color_sequence)
    fig.update_traces(textposition="top center", marker=dict(size=12))
    return style_figure(fig, x_reverse=x_reverse)
