import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

#######
# Objective: build a dashboard that imports OldFaithful.csv
# from the data directory, and displays a scatterplot.
# The field names are:
# 'D' = date of recordings in month (in August),
# 'X' = duration of the current eruption in minutes (to nearest 0.1 minute),
# 'Y' = waiting time until the next eruption in minutes (to nearest minute).
######

app = dash.Dash() # Create a flask application

# Creating data
df = pd.read_csv('data/OldFaithful.csv')

# Create a plotly graph inside dash
app.layout = html.Div([dcc.Graph(id='scatterplot',
                        figure = {'data':[
                                    go.Scatter(
                                    x=df['X'],
                                    y=df['Y'],
                                    mode='markers'
                                    )],
                                'layout':
                                    go.Layout(title="OldFaithful",
                                            xaxis= {'title':'duration of the current eruption in minutes'},
                                            yaxis= {'title':'waiting time until the next eruption in minutes'})}
                    )])

if __name__ == '__main__':
    app.run_server()
