# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:25:03 2023

@author: jathi
"""
import sklearn
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import dash
import plotly.express as px
from dash import dcc, html, Dash
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import joblib
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
dbc_css = 'https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css'

excel_file_path = 'selected_columns.xlsx'  # Change this to your file path and name

# Use the read_excel function to read data from the Excel file into a DataFrame
selected_columns = pd.read_excel(excel_file_path)

# Define the feature columns
feature_columns = ['Alcoholintakeg/d','Dietaryfiberg/d', 'Calciummg/d', 'Totalenergykcal/d',
                   'Carbohydrateg/d', 'Cholesterolmg/d', 'Proteing/d', 'Totalfatg/d',
                   'Alpha-Tocopherolmg/d', 'Ironmg/d', 'Glycemicindex', 'Glycemicloadg/d',
                   'Gamma-Tocopherolmg/d', 'Monounsaturatedfatg/d', 'Omega-6fattyacidsg/d',
                   'Omega-3fattyacidsg/d', 'Polyunsaturatedfatg/d', 'Saturatedfatg/d', 'Transfatg/d','age','sex','smokes','exercise']

# Split the data into features and target variable
df = selected_columns[feature_columns]

dropdown = dcc.Dropdown(
    id='column-dropdown',
    options=[{'label': col, 'value': col} for col in df.columns],
    value=list(df.columns),  # Default selected column as a list
    multi=True,
    style={'width': '100%'}
)

description = """
Impact of diet on Blue zone population.
Data is extracted from [CRELES](https://www.norges-bank.no/en/topics/Statistics/exchange_rates/?tab=api)"""

card_header = dbc.Card(
    children = [
        html.H1('BLUE ZONES'),
        dcc.Markdown(description),
    ],
    
    body = True
)

tab1_content = dbc.Container(
    children=[
        html.Br(),
        html.P(
            children="Step 1: Select an Input Variable",
            style={'textAlign': 'center'}
        ),
        dropdown,  # Dropdown for variable selection
        html.Br(),  # Add some space
        html.P(
            children="Step 2: Explore the Distribution",
            style={'textAlign': 'center'}
        ),
        #dcc.Graph(id='histogram-plot'),  # Hist plot
        html.Div(id='histogram-plot'),
    ]
)

tab2_content = dbc.Container(
    children=[
        html.H1('Hypertension Predictor', className='mb-4'),

        dbc.Row([
            dbc.Col(html.Label('Servings of Carbs:', className='mb-2'), width=3),
            dbc.Col(dcc.Input(id='servings-carbs-input', type='number', value=0), width=9),
        ], className='mb-3'),

        dbc.Row([
            dbc.Col(html.Label('Servings of Nuts:', className='mb-2'), width=3),
            dbc.Col(dcc.Input(id='servings-nuts-input', type='number', value=0), width=9),
        ], className='mb-3'),

        dbc.Row([
            dbc.Col(html.Label('Servings of Vegetables:', className='mb-2'), width=3),
            dbc.Col(dcc.Input(id='servings-vegetables-input', type='number', value=0), width=9),
        ], className='mb-3'),

        dbc.Row([
            dbc.Col(html.Label('Servings of Proteins:', className='mb-2'), width=3),
            dbc.Col(dcc.Input(id='servings-proteins-input', type='number', value=0), width=9),
        ], className='mb-3'),

        dbc.Row([
            dbc.Col(html.Label('Servings of Fruits:', className='mb-2'), width=3),
            dbc.Col(dcc.Input(id='servings-fruits-input', type='number', value=0), width=9),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col(html.Label('Sex:', className='mb-2'), width=3),
            dbc.Col(dcc.Dropdown(
                id='sex-input',
                options=[
                    {'label': 'Male', 'value': 1},
                    {'label': 'Female', 'value': 0},
                ],
                value=1,  # Default value for Male
            ), width=3),
        ], className='mb-3'),

        dbc.Row([
            dbc.Col(html.Label('Smokes:', className='mb-2'), width=3),
            dbc.Col(dcc.Dropdown(
                id='smokes-input',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0},
                ],
                value=1,  # Default value for Yes
            ), width=3),
        ], className='mb-3'),

        dbc.Row([
            dbc.Col(html.Label('Exercise:', className='mb-2'), width=3),
            dbc.Col(dcc.Dropdown(
                id='exercise-input',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0},
                ],
                value=1,  # Default value for Yes
            ), width=3),
        ], className='mb-3'),

        dbc.Row([
            dbc.Col(html.Div(id='prediction-output'), width=12),
        ], className='mb-3'),

        dbc.Row([
            dbc.Col(html.Button('Predict', id='predict-button', n_clicks=0, className='btn btn-primary'), width={'size': 6, 'offset': 3}),
        ], className='mb-3')
    ],
    className='mt-5'
)





# Sort the DataFrame by 'age' column in descending order
sorted_df = df.sort_values(by='age', ascending=False)

# Format numerical columns to two decimal places
formatted_df = sorted_df.apply(lambda x: round(x, 2) if pd.api.types.is_numeric_dtype(x) else x)

# Ensure 'Age' is the first column in the DataFrame
if 'age' in formatted_df:
    formatted_df = formatted_df[['age'] + [col for col in formatted_df if col != 'age']]

# Create a container with an internal scroll bar for the table
table_container = html.Div(
    dbc.Table.from_dataframe(formatted_df, striped=True, bordered=True, hover=True),
    style={'overflowY': 'scroll', 'maxHeight': '400px'}  # Adjust maxHeight to your preference
)

tab3_content = dbc.Container(
    children=[
        html.Br(),
        table_container
    ]
)


app = Dash(__name__, external_stylesheets = [dbc.themes.JOURNAL, dbc_css])

# Load the best classifier
best_classifier = joblib.load('best_classifier.pkl')

app.layout = dbc.Container(
    children = [
        
        # Header
        card_header,
        html.Br(),
        
        # Main tabs
        dbc.Card(dbc.Tabs(
           children = [
                dbc.Tab(tab1_content, label = 'Explore'),
                dbc.Tab(tab2_content, label = 'Predict'),
                dbc.Tab(tab3_content, label = 'Data'),
           ]
       ), body = True)
        
    ],
    
    className = 'dbc'
)

# Define callback to update the histogram based on multi-select dropdown selection
@app.callback(
    Output('histogram-plot', 'children'),
    Input('column-dropdown', 'value')
)



def update_histogram(selected_columns):
    if not selected_columns:
        raise dash.exceptions.PreventUpdate

    selected_columns = ['Alcoholintakeg/d', 'Dietaryfiberg/d', 'Calciummg/d', 'Totalenergykcal/d',
                   'Carbohydrateg/d', 'Cholesterolmg/d', 'Proteing/d', 'Totalfatg/d',
                   'Alpha-Tocopherolmg/d', 'Ironmg/d', 'Glycemicindex', 'Glycemicloadg/d',
                   'Gamma-Tocopherolmg/d', 'Monounsaturatedfatg/d', 'Omega-6fattyacidsg/d',
                   'Omega-3fattyacidsg/d', 'Polyunsaturatedfatg/d', 'Saturatedfatg/d', 'Transfatg/d']

    # Calculate the number of rows and columns for subplots
    num_columns = len(selected_columns)
    num_rows = (num_columns + 1) // 2  # 2 subplots per row

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

    # Flatten the 2D array of subplots for easier indexing
    axs = axs.flatten()

    # Define a custom color palette for the legend
    sex_colors = {0: 'blue', 1: 'red'}

    # Loop through selected columns and create separate histograms
    for i, column in enumerate(selected_columns):
        ax = axs[i]

        # Create histograms based on sex
        for sex, color in sex_colors.items():
            sns.histplot(df[df['sex'] == sex][column], bins=20, kde=True, ax=ax, color=color, label=f'Sex {sex}')

        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')

        # Add legend to each subplot
        ax.legend(title='Sex', labels=['Female', 'Male'])

    # Remove any extra empty subplots
    for i in range(num_columns, len(axs)):
        fig.delaxes(axs[i])

    # Adjust subplot spacing
    plt.tight_layout()

    # Save the figure as an image
    image_filename = "histograms.png"
    fig.savefig(image_filename)

    # Encode the image as base64
    with open(image_filename, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Return the image as an HTML Img component
    return html.Img(src=f"data:image/png;base64,{encoded_image}", style={'width': '100%'})

# Define a callback to update model inputs and perform prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('servings-carbs-input', 'value'),
    Input('servings-nuts-input', 'value'),
    Input('servings-vegetables-input', 'value'),
    Input('servings-proteins-input', 'value'),
    Input('servings-fruits-input', 'value'),
    Input('sex-input', 'value'),
    Input('smokes-input', 'value'),
    Input('exercise-input', 'value'),
)
def predict_hypertension(n_clicks, servings_carbs, servings_nuts, servings_vegetables, servings_proteins, servings_fruits, sex, smokes, exercise):
    try:
        if n_clicks > 0:
            # Calculate model inputs from servings and other user inputs
            input_data = calculate_servings(servings_carbs, servings_nuts, servings_vegetables, servings_proteins, servings_fruits)
            input_data['sex'] = [sex]  # Wrap scalar value in a list
            input_data['smokes'] = [smokes]  # Wrap scalar value in a list
            input_data['exercise'] = [exercise]  # Wrap scalar value in a list

            # Create a DataFrame from the input data
            input_df = pd.DataFrame(input_data)

            # Predict hypertension probability
            hypertension_probability = best_classifier.predict_proba(input_df)[0][1]

            # Display the prediction result
            result_text = f"Predicted Hypertension Probability: {hypertension_probability:.2f}"
            return result_text
    except Exception as e:
        return str(e)

    return ""

if __name__ == '__main__':
    app.run_server(debug = True)
