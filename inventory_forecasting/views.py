# Create your views here.

import numpy as np
import pandas as pd
# from .forecasting import run_forecasting_code 
from django.shortcuts import render, HttpResponse
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from inventory_forecasting.models import Inventory_Item

# Create your views here.

def index(request):
    return HttpResponse("Heyyyyyyyyyyyyyyyyyy Its me mY VOy")

def func(request):
    return HttpResponse("Heyyyyyyyyyyyyyyyyyyyyyyyy This is really Fun")


# Add Item Stock To Inventory
def addItemStock(request):
    if request.method == 'POST':
        item_name_req = request.POST.get('item name')
        item_quantity_req = request.POST.get('add quantity')
        if Inventory_Item.objects.filter(item_name=item_name_req).exists():
            obj_to_update = Inventory_Item.objects.get(item_name=item_name_req)
            obj_to_update.item_quantity = obj_to_update.item_quantity + int(item_quantity_req)
            obj_to_update.date = datetime.today()
            obj_to_update.save()
        else:
            inventory_item = Inventory_Item(item_name = item_name_req, item_quantity = item_quantity_req, date = datetime.today())
            inventory_item.save()
    return render(request, 'forecasting_result.html')


# Order Item Stock From Inventory
def orderItemStock(request):
    if request.method == 'POST':
        item_name_req = request.POST.get('item name')
        item_quantity_req = int(request.POST.get('order quantity'))
        if Inventory_Item.objects.filter(item_name=item_name_req).exists():
            obj_to_update = Inventory_Item.objects.get(item_name=item_name_req)
            if obj_to_update.item_quantity < item_quantity_req:
                return render(request, 'forecasting_result.html', {"message" : "Inventory doesn't have that much quantity of Item"})
            obj_to_update.item_quantity = obj_to_update.item_quantity - item_quantity_req
            obj_to_update.save()
        else:
            return render(request, 'forecasting_result.html', {"message" : "Item doestn't exist in Inventory"})
    return render(request, 'forecasting_result.html')





# inventory_forecasting/views.py


# inventory_forecasting/views.py

def inventory_forecasting(request):
    # Path to your local Excel file
    excel_file_path = 'C:\Django Project\mysite\static\inventoryData.csv'

    try:
        # Load your dataset
        df = pd.read_csv(excel_file_path, parse_dates=['Date'], date_format='%m-%d-%Y')
        df_dummies = pd.get_dummies(df, columns= ['Product line'])
        df_final = pd.concat([df, df_dummies], axis=1, join='inner')
        # Feature Engineering: Extract relevant date features
        df_final['Year'] = df['Date'].dt.year
        df_final['Month'] = df['Date'].dt.month
        df_final.drop(['Product line', 'Quantity', 'Date'], axis=1, inplace= True)
        # Split the data into training and test sets
        X = df_final  # Features
        y = df['Quantity']  # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        # Create and train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # Show Pull data from DB here 
        
        futureQuanitity = prepare_results(model)
        chart_image = generate_chart(y_test, y_pred)

        # Pass the results to a template and render it
        return render(request, 'forecasting_result.html', {'results': futureQuanitity, 'chart_image': chart_image
                                                       })

    except Exception as e:
        return render(request, 'error.html', {'error_message': str(e)})
    
def prepare_results(model):
    results={}
    results['Product line_Electronic accessories'] = model.predict(create_data_frame(Electronic_accessories=True))
    results['Product line_Fashion accessories'] = model.predict(create_data_frame(Fashion_accessories=True))
    results['Product line_Food and beverages'] = model.predict(create_data_frame(Food_and_beverages=True))
    results['Product line_Health and beauty'] = model.predict(create_data_frame(Health_and_beauty=True))
    results['Product line_Home and lifestyle'] = model.predict(create_data_frame(Home_and_lifestyle=True))
    results['Product line_Sports and travel'] = model.predict(create_data_frame(Sports_and_travel=True))
    return results


def create_data_frame(Electronic_accessories = False, Fashion_accessories = False, Food_and_beverages = False,
                  Health_and_beauty = False, Home_and_lifestyle = False, Sports_and_travel = False,
                  Year = '2015', Month = '10'):
    df = pd.DataFrame()
    df['Product line_Electronic accessories'] = [Electronic_accessories]
    df['Product line_Fashion accessories']    = [Fashion_accessories]
    df['Product line_Food and beverages']     = [Food_and_beverages]
    df['Product line_Health and beauty']      = [Health_and_beauty]
    df['Product line_Home and lifestyle']     = [Home_and_lifestyle]
    df['Product line_Sports and travel']      = [Sports_and_travel]
    df['Year']                                = [Year]
    df['Month']                               = [Month]
    return df

def generate_chart(y_test, y_pred):
    # Generate a Matplotlib chart
    plt.figure(figsize=(5, 5))
    plt.plot(y_test, y_pred)
    plt.ylabel('Predicted Quantity')
    plt.xlabel('Actual Quantity')
    plt.title('Actual Quantity VS Predicted Quantity')

    # Save the chart as a base64-encoded image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_image = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return chart_image
