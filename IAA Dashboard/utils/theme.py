import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import dash_bootstrap_components as dbc
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from ..app import app

@app.callback(
    [Output('theme-store', 'data'),
     Output('light-theme-btn', 'color'),
     Output('dark-theme-btn', 'color')],
    [Input('light-theme-btn', 'n_clicks'),
     Input('dark-theme-btn', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(light_clicks, dark_clicks, current_theme):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'light', 'primary', 'outline-secondary'
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'light-theme-btn':
        return 'light', 'primary', 'outline-secondary'
    elif trigger_id == 'dark-theme-btn':
        return 'dark', 'outline-secondary', 'primary'
    
    return current_theme, 'primary' if current_theme == 'light' else 'outline-secondary', 'primary' if current_theme == 'dark' else 'outline-secondary'