import os
import numpy as np
import pandas as pd
from django.shortcuts import render, HttpResponse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
from inventory_forecasting.models import Inventory_Item

# Add Item Stock To Inventory
def addItemStock(request):
    items = Inventory_Item.objects.all()
    item_data ={}
    for item in items:
        item_data[item.item_id] = [item.item_name, item.item_quantity]
    response = {
        "item_data" : item_data
    }
    success_message = ""
    if request.method == 'POST':
        item_id_req = request.POST.get('item_id')
        item_name_req = request.POST.get('item name')
        item_quantity_req = request.POST.get('add quantity')
        if Inventory_Item.objects.filter(item_id=item_id_req).exists():
            obj_to_update = Inventory_Item.objects.get(item_id=item_id_req)
            obj_to_update.item_quantity = obj_to_update.item_quantity + int(item_quantity_req)
            obj_to_update.item_name=item_name_req
            obj_to_update.date = datetime.today()
            obj_to_update.save()
            success_message=f"Stock added to the existing stock for Item : {item_name_req}"
            response["success_message"] = success_message
        else:
            inventory_item = Inventory_Item(item_id = item_id_req,item_name = item_name_req, item_quantity = item_quantity_req, date = datetime.today())
            inventory_item.save()
            success_message=f"Stock created for Item : {item_name_req}"
            response["success_message"] = success_message
    return render(request, 'forecasting_result.html', response)


# Order Item Stock From Inventory
def showInventoryStock(request):
    items = Inventory_Item.objects.all()
    tot_data = []
    for item in items:
        item_data = {
            "item_id" : item.item_id,
            "item_name" : item.item_name,
            "item_quantity" : item.item_quantity,
            "date" : item.date
        }
        tot_data.append(item_data)
    print(tot_data)
    return render(request, 'showItems.html', {"tot_data" : tot_data})

# Order Item Stock From Inventory
def orderItemStock(request):
    items = Inventory_Item.objects.all()
    item_data ={}
    for item in items:
        item_data[item.item_id] = [item.item_name, item.item_quantity]
    if request.method == 'POST':
        item_id_req = request.POST.get('item_id')
        item_quantity_req = int(request.POST.get('order quantity'))
        if Inventory_Item.objects.filter(item_id=item_id_req).exists():
            obj_to_update = Inventory_Item.objects.get(item_id=item_id_req)
            if obj_to_update.item_quantity < item_quantity_req:
                return render(request, 'forecasting_result.html', {"error_message_new" : f"Stock Shortfall for Item : {obj_to_update.item_name}",
                                                                   "item_data" : item_data})
            obj_to_update.item_quantity = obj_to_update.item_quantity - item_quantity_req
            obj_to_update.save()
        else:
            return render(request, 'forecasting_result.html', {"error_message_new" : "Item doestn't exist in Inventory",
                                                               "item_data" : item_data})
    return render(request, 'forecasting_result.html', {
        "item_data" : item_data
    })



def inventory_forecasting(request):
    file_name = request.POST.get('predict')
    error_message=""
    chart_image1=""
    df=""
    excel_file_path = f'C:\Django Project\mysite\static\{file_name}.xlsx'
    if os.path.exists(excel_file_path):
        try:
            df = pd.read_excel(excel_file_path)
        except Exception as e:
            error_message = f"Error reading the file: {str(e)}"
    else:
        error_message = f"{file_name} data doesn't exist for Analysis"
                                                  
    if type(df) != str:
        df_final = df.copy(deep=True)
        df_final.drop(['2022 prediction'], axis=1, inplace= True)
        X = df_final  # Features
        y = df['2022 prediction']  # Target variable

        X.columns = X.columns.astype(str)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(df_final)
        # Calculate squared differences
        squared_differences = (y - y_pred) ** 2

        # Calculate MSE
        mse = np.mean(squared_differences)

        print("Mean Squared Error (MSE):", mse)
        chart_image1 = generate_chart(df_final, y_pred)
    items = Inventory_Item.objects.all()
    item_data ={}
    for item in items:
        item_data[item.item_id] = [item.item_name, item.item_quantity]

    response = {
                'item_data' : item_data
                }
    if len(chart_image1) != 0:
        response['chart_image'] = chart_image1
    if len(error_message) != 0:
        response['error_message'] = error_message
    # Pass the results to a template and render it
    return render(request, 'forecasting_result.html', response)
    

def generate_chart(df, y_pred):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    #years = list(range(2013, 2023))
    years = list(df.columns)
    years = years[1:]
    print(years)
    data = {}
    for year in years:
        data[year] = df[year].tolist()

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each year's data
    for year in years:
        plt.plot(months, data[year], label=str(year))
    plt.plot(months, y_pred, label ="2022 Predicted")
    plt.plot(months, [85,85,85,85,85,85,85,85,85,85,85,85], label = "static threshold")
    # Add labels and legend
    plt.xlabel("Month")
    plt.ylabel("Quantity")
    plt.title("Data Over Time")
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_image = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return chart_image


def generate_chart_new(y_actual, y_pred):
    # Create the figure and axis
    plt.subplots(figsize=(12, 6))
    plt.plot(y_actual, y_pred)
    # Add labels and legend
    plt.xlabel("Quantity Actual")
    plt.ylabel("Quantity Predicted")
    plt.title("Actual Quantity Vs Predicted Quantity")
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_image = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return chart_image