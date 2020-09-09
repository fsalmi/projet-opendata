import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pathlib
import os
import pandas as pd
import numpy as np

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.config["suppress_callback_exceptions"] = True

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
df = pd.read_csv("Predictions_test.csv")



#markdown_text1 = '''
##### Prediction of electricity consumption 
#We used a [Dataset](https://opendata.reseaux-energies.fr/explore/embed/dataset/consommation-quotidienne-brute-elec/table/?disjunctive.qualite&sort=date_heure&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJsaW5lIiwiZnVuYyI6IkFWRyIsInlBeGlzIjoiY29uc29tbWF0aW9uIiwiY29sb3IiOiIjZWE1MjU0Iiwic2NpZW50aWZpY0Rpc3BsYXkiOnRydWV9XSwieEF4aXMiOiJkYXRlX2hldXJlIiwibWF4cG9pbnRzIjpudWxsLCJ0aW1lc2NhbGUiOiJob3VyIiwic29ydCI6IiIsImNvbmZpZyI6eyJkYXRhc2V0IjoiY29uc29tbWF0aW9uLXF1b3RpZGllbm5lLWJydXRlLWVsZWMiLCJvcHRpb25zIjp7ImRpc2p1bmN0aXZlLnF1YWxpdGUiOnRydWV9fX1dLCJ0aW1lc2NhbGUiOiJ5ZWFyIiwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZX0%3D) 
#containing the French electricity consumption with a 30 minutes step, for a national level. 
#The aim of the project was to predict the electricity consumption with a high accuracy. The method used for this purpose was first to extract the trend and seasonality of the time series and then to predict the trend with a LSTM neural network and the seasonality with a Fourier decomposition. 
#Below, we see the prediction for the 2019 year with a sliding window to check day, week, month values. This model was trained on five years from 2014 to 2018 and the predictions reach an accuracy of 96%. 
#
#'''

markdown_text1 = '''
### 
Nous avons utilisé un jeu de [données](https://opendata.reseaux-energies.fr/explore/embed/dataset/consommation-quotidienne-brute-elec/table/?disjunctive.qualite&sort=date_heure&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJsaW5lIiwiZnVuYyI6IkFWRyIsInlBeGlzIjoiY29uc29tbWF0aW9uIiwiY29sb3IiOiIjZWE1MjU0Iiwic2NpZW50aWZpY0Rpc3BsYXkiOnRydWV9XSwieEF4aXMiOiJkYXRlX2hldXJlIiwibWF4cG9pbnRzIjpudWxsLCJ0aW1lc2NhbGUiOiJob3VyIiwic29ydCI6IiIsImNvbmZpZyI6eyJkYXRhc2V0IjoiY29uc29tbWF0aW9uLXF1b3RpZGllbm5lLWJydXRlLWVsZWMiLCJvcHRpb25zIjp7ImRpc2p1bmN0aXZlLnF1YWxpdGUiOnRydWV9fX1dLCJ0aW1lc2NhbGUiOiJ5ZWFyIiwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZX0%3D) 
provenant des données publiques de consommations annuelles d'électricité avec un pas de 30 minutes, au niveau national. 
Le but du projet était de prédire avec une grande précision la consommation annuelle d'électricité à l'échelle nationale avec une granularité de 30 minutes.  
La méthode suivie a été, dans un premier temps, d'extraire la tendance et la saisonnalité de la série temporelle, puis dans un second temps de traiter ces deux composantes 
séparément et enfin de recombiner les deux prédictions. La tendance a été prédite à l'aide d'un réseau de neurones LSTM et la saisonnalité a été déterminée à l'aide d'une décomposition de Fourier. 

Ci-dessous, est représentée la prédiction pour l'année 2019 avec une fenêtre glissante qui montre les valeurs avec une échelle d'une journée, d'une semaine ou d'un mois. 
Le modèle a été entraîné sur une période de cinq ans (de 2014 à 2018) et les prédictions sur l'année 2019 atteignent une précision de 96%. 

Cette étude pourra être améliorée en tenant compte des conditions météorologiques qui ont un impact fort sur la consommation d'électricité. De plus le même type de données pour certaines régions seront rendues publiques d'ici la fin de l'année. 
Il sera alors possible de mener la même étude avec une granularité spatiale plus fine.

'''

markdown_text2 = '''
#### Insight on the database used  

'''






colors = {
    'background': 'black',
    'text': 'white'
}

def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Prédiction de la consommation d'électricité"),
                    
                ],
            ),
            
            html.Div(
                id="banner-logo",
                children=[
                    #html.Button(
                    #    id="learn-more-button", children="LEARN MORE", n_clicks=0
                    #),
                    html.Img(id="logo_cap", src=app.get_asset_url("logo_Capgemini.jpg")),
                ],
            ),
        ],
    )


app.layout = html.Div(
        id="big-app-container",
        children=[
            build_banner(),
            # header
            html.Div([
            dcc.Markdown(children=markdown_text1)
            ],style= {'color':'white'}),
            html.Div(html.P([html.Br()])),

            html.Div([
            dcc.Dropdown(id='yaxis',
                         options = [
                                    {'label': 'Prédiction de la consommation d\'électricité en utilisant un LSTM (tendance) et une décomposition de Fourier (saisonnalité)', 'value': 'Predictions'}],
                                    #{'label': 'Prediction using trend (LSTM) and seasonality (Fourier)', 'value': 'Predictions'}],
                         value = 'Predictions')#
                         
            ],style={'width':'100%'}),#
           
            
            dcc.Graph(id='feature-graphics')])

           
            
@app.callback(Output('feature-graphics','figure'),
[Input('yaxis','value')])

def update_graph(yaxis_column_name):
    traces = []
    #print(df['Valeurs'])
    traces.append(dict(
            x=df['day_time'],
            y=df['Valeurs'],
            #text=df_by_continent['country'],
            mode='lines',
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Valeurs'
        ))
    #print(df[yaxis_column_name])
    traces.append(dict(
            x=df['day_time'],
            y=df[yaxis_column_name],
            mode='lines',
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Prédictions'
        ))



    return {
        'data': traces ,
        
        'layout':go.Layout(
                xaxis=dict(title='Date',
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                label="Jour",
                                step="day",
                                stepmode="backward"),
                            dict(count=7,
                                label="Semaine",
                                step="day",
                                stepmode="backward"),  
                            dict(count=1,
                                label="Mois",
                                step="month",
                                stepmode="backward"), 
                            dict(label="Total",step="all")
                        ])
                ),
                rangeslider=dict(
                visible=True
                ),
                type="date"
                ),
                yaxis={'title': 'Consommation d\'électricité (MW)'},
                height=700, # px
                width=1150
    )}






if __name__ == '__main__':
    app.run_server(debug=False, port=5000)    
    #app.run_server(debug=True, host='0.0.0.0',port=8053)