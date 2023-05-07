# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

 # Load dataset
data = pd.read_csv('data/winequality-red.csv')
# Check for missing values
data.isna().sum()
# Remove duplicate data
data.drop_duplicates(keep='first')
# Calculate the correlation matrix
corr_matrix = data.corr()
# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)
# Drop the target variable
X = data.drop('quality', axis=1)
# Set the target variable as the label
y = data['quality']


# Split the dat a into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Create an instance of the logistic regression model
logreg_model = LogisticRegression()
# Fit the model to the training data
logreg_model.fit(X_train, y_train)

# Dash Application
# Create the Dash app
app = dash.Dash(__name__)

server = app.server

# Define the layout of the dashboard
app.layout = html.Div(

  children=[
      html.Div([
           html.H1('CO544-2023 Lab 3: Wine Quality Prediction', style={"text-align": "center"})
      ]),

      html.Div([
          html.H2('Exploratory Data Analysis',  style={"text-align": "center" }),
      ]),

  # Layout for exploratory data analysis: correlation between two selected features
  # Center the two Dropdowns, make them easy to lacate
  html.Div([
      html.Div([
      
      html.Div([
          html.Label('Feature 1 (X-axis)', style={'font-size': 20}),
      ], style={'margin-bottom': '10px'}),
      
      # Dropdown menu to select one feature
      dcc.Dropdown(
        id='x_feature',
        options=[{'label': col, 'value': col, } for col in data.columns],
        value=data.columns[0],
        style={'font-size': 20}
      )
    ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '100px'}),

      
      html.Div([
        html.Div([
          html.Label('Feature 2 (Y-axis)', style={'font-size': 20}),
      ], style={'margin-bottom': '10px'}),

        # Dropdown menu to select other feature
        dcc.Dropdown(
          id='y_feature',
          options=[{'label': col, 'value': col} for col in data.columns],
          value=data.columns[1],
          style={'font-size': 20}
      )
      ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '100px'}),
  ], style={'justify-content': "center", 'display': 'flex', 'margin-top' : '10px'}),

  

  # Graph id
  dcc.Graph(id='correlation_plot'),


  # Layout for wine quality prediction based on input feature values
  html.H2("Wine Quality Prediction", style={"text-align": "center" }),

  # html.Div([
    
    html.Div([
      html.Div([
        html.Label("Fixed Acidity", style={'font-size': 20}, ),
        dcc.Input(id='fixed_acidity', type='number', required=True, style={
              'font-size': '15px',
              'padding': '8px',
              'border': '0.8px solid #ccc',
              'border-radius': '1px',
              'width': '100px',
              'margin-left': '10px',
              'text-align': 'center',
          }),
      ],  style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'}),

      html.Div([
        html.Label("Volatile Acidity",  style={'font-size': 20}),
        dcc.Input(id='volatile_acidity', type='number', required=True, style={
              'font-size': '15px',
              'padding': '8px',
              'border': '0.8px solid #ccc',
              'border-radius': '1px',
              'width': '100px',
              'margin-left': '10px',
              'text-align': 'center'
          }),
      ],  style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'}),
      
      html.Div([
        html.Label("Citric Acid", style={'font-size': 20}),
        dcc.Input(id='citric_acid', type='number', required=True, style={
              'font-size': '15px',
              'padding': '8px',
              'border': '0.8px solid #ccc',
              'border-radius': '1px',
              'width': '100px',
              'margin-left': '10px',
              'text-align': 'center'
          }),
      ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'}),

    ],style={'justify-content': "center", 'display': 'flex', 'margin-top' : '20px'}),


    html.Div([

      html.Div([
        html.Label("Residual Sugar", style={'font-size': 20}),
        dcc.Input(id='residual_sugar', type='number', required=True, style={
                'font-size': '15px',
                'padding': '8px',
                'border': '0.8px solid #ccc',
                'border-radius': '1px',
                'width': '100px',
                'margin-left': '10px',
                'text-align': 'center'
            }),
      ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'}),

      html.Div([
        html.Label("Chlorides", style={'font-size': 20}),
        dcc.Input(id='chlorides', type='number', required=True, style={
              'font-size': '15px',
              'padding': '8px',
              'border': '0.8px solid #ccc',
              'border-radius': '1px',
              'width': '100px',
              'margin-left': '10px',
              'text-align': 'center'
          }),
      ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'}),

      html.Div([
        html.Label("Free Sulfur Dioxide", style={'font-size': 20}),
        dcc.Input(id='free_sulfur_dioxide', type='number', required=True, style={
              'font-size': '15px',
              'padding': '8px',
              'border': '0.8px solid #ccc',
              'border-radius': '1px',
              'width': '100px',
              'margin-left': '10px',
              'text-align': 'center'
          }),
      ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'})

    ], style={'justify-content': "center", 'display': 'flex', 'margin-top' : '10px'}),


    html.Div([

      html.Div([
        html.Label("Total Sulfur Dioxide", style={'font-size': 20}),
        dcc.Input(id='total_sulfur_dioxide', type='number', required=True, style={
              'font-size': '15px',
              'padding': '8px',
              'border': '0.8px solid #ccc',
              'border-radius': '1px',
              'width': '100px',
              'margin-left': '10px',
              'text-align': 'center'
          }),
      ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'}),

      html.Div([
        html.Label("Density", style={'font-size': 20}),
        dcc.Input(id='density', type='number', required=True, style={
              'font-size': '15px',
              'padding': '8px',
              'border': '0.8px solid #ccc',
              'border-radius': '1px',
              'width': '100px',
              'margin-left': '10px',
              'text-align': 'center'
          }),
      ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'}),

      html.Div([
        html.Label("pH", style={'font-size': 20}),
        dcc.Input(id='ph', type='number', required=True, style={
              'font-size': '15px',
              'padding': '8px',
              'border': '0.8px solid #ccc',
              'border-radius': '1px',
              'width': '100px',
              'margin-left': '10px',
              'text-align': 'center'
          }),
      ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'})

    ], style={'justify-content': "center", 'display': 'flex', 'margin-top' : '10px'}),


    html.Div([


      html.Div([
        html.Label("Sulphates", style={'font-size': 20}),
        dcc.Input(id='sulphates', type='number', required=True, style={
              'font-size': '15px',
              'padding': '8px',
              'border': '0.8px solid #ccc',
              'border-radius': '1px',
              'width': '100px',
              'margin-left': '10px',
              'text-align': 'center'
          }),
      ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'}),

      html.Div([
        html.Label("Alcohol", style={'font-size': 20}),
        dcc.Input(id='alcohol', type='number', required=True, style={
              'font-size': '15px',
              'padding': '8px',
              'border': '0.8px solid #ccc',
              'border-radius': '1px',
              'width': '100px',
              'margin-left': '10px',
              'text-align': 'center'
          }),
      ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '50px'}),

    ], style={'justify-content': "left", 'display': 'flex', 'margin-top' : '10px'}),
    
  # ], style={'display': 'flex', 'flexWrap': 'wrap'}),

# Add prediction Button
  html.Div([
    html.Button('Predict', id='predict-button', n_clicks=0, style={'width': '150px', 'height': '35px', 'font-size': '25px'}),
  ], style={'justify-content': "center", 'display': 'flex', 'margin-top' : '30px'}),

# Showing predicted quality
  html.Div([
    html.H2("Predicted Quality"),
  ], style={'justify-content': "center", 'display': 'flex', 'margin-top' : '10px'}),

  # Output
  html.Div([
    html.Div(
      html.H2(id='prediction-output', style={"text-align": "center", 'color': 'red'})
    )
  ], style={'justify-content': "center", 'display': 'flex', 'margin-top' : '5px'})
])


# Interactivity
# Define the callback to update the correlation plot
# When the x feature or Y feature is changed, correlation plot is updated
import numpy as np
@app.callback(
dash.dependencies.Output('correlation_plot', 'figure'),
  [dash.dependencies.Input('x_feature', 'value'),
  dash.dependencies.Input('y_feature', 'value')]
)

# Method to update correlation plot
def update_correlation_plot(x_feature, y_feature):
  # Save the scatter plot to the fig variable
  fig = px.scatter(data, x=x_feature, y=y_feature, color='quality', color_continuous_scale='bluered')
  # Update the title of the figure
  fig.update_layout(title=f"Correlation between {x_feature} and {y_feature}")
  # Return the updated figure object
  return fig


# Define the callback function to predict wine quality
@app.callback(
  # Children of the prediction-output id will be updated as the output of the callback function
  Output(component_id='prediction-output', component_property='children'),
  # Callback function is triggered when predict-button is pressed
  [Input('predict-button', 'n_clicks')],

  # State objects with value properties and ids
  [State('fixed_acidity', 'value'),
  State('volatile_acidity', 'value'),
  State('citric_acid', 'value'),
  State('residual_sugar', 'value'),
  State('chlorides', 'value'),
  State('free_sulfur_dioxide', 'value'),
  State('total_sulfur_dioxide', 'value'),
  State('density', 'value'),
  State('ph', 'value'),
  State('sulphates', 'value'),
  State('alcohol', 'value')]
)

def predict_quality(n_clicks, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol):
  # Create input features array for prediction
  input_features = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]).reshape(1, -1)
  # Predict the wine quality (0 = bad, 1 = good)
  prediction = logreg_model.predict(input_features)[0]

  # Return the prediction
  if prediction == 1:
    return "This wine is predicted to be good quality."
  else:
    return "This wine is predicted to be bad quality."

# Run the dash application 
if __name__ == '__main__':
  app.run_server(debug=False)