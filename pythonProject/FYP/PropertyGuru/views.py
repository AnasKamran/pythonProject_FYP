import pickle
import random

from django.http import HttpResponseRedirect
from .forms import forms
from django.db.models import Q
import csv
import math
from django.shortcuts import render, get_object_or_404
from .models import Feedback,Properties
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KDTree
import warnings
import re
import locale
import matplotlib.pyplot as plt
import os
import json
import requests
warnings.simplefilter("ignore")

def about(request):
    return render(request,'about.html')

def nearbyproperties(request):
    return render(request,'nearbyproperties.html')
def Errors(request):
    return render(request,'Error.html')

def contact(request):
    if request.method == 'POST':
        form = forms(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            E_mail = form.cleaned_data['E_mail']
            ph_no = form.cleaned_data['ph_no']
            subject = form.cleaned_data['subject']
            Details = form.cleaned_data['Details']
            form.save()
            return HttpResponseRedirect('/forms/')
    else:
        form = forms()

    return render(request, 'contact.html', {'form': form})


def search(request):
    if 'search_term' in request.GET:
        search_term = request.GET['search_term']
        purpose = ""
        try:
            purpose = request.GET['radio_button']
        except Exception as e:
            print(e)
        if purpose == "Sale" or purpose == "Rent":
            properties = Properties.objects.filter(purpose__icontains=purpose, address__icontains=search_term)
        else:
            properties = Properties.objects.filter(address__icontains=search_term)
        count = properties.count()

        recommendation = Properties.objects.filter(address__icontains=search_term).exclude(image_src__exact='').order_by('?')[:6]
        if count == 0:
            recommendation = Properties.objects.filter(purpose__icontains='Rent').exclude(image_src__exact='').order_by('?')[:6]

    else:
        properties = Properties.objects.none()
        recommendation = Properties.objects.none()
        search_term = ""
        count = 0
        purpose = ""
    context = {
        'properties': properties,
        'search_term': search_term,
        'count': count,
        'purpose': purpose,
        'recommendation': recommendation
    }

    return render(request, 'available properties.html', context)


def index (request):
    all_property_ids = Properties.objects.values_list('Property_id', flat=True)
    property_id = random.choice(all_property_ids)
    recommend = recommendations(property_id)
    list=[]
    for i in recommend:
        list.append(i)
    recommendation = Properties.objects.filter(Property_id__in=list)
    # recommendation = Properties.objects.exclude(image_src__isnull=True).order_by('?')[:10]
    return render(request, 'index.html', {'recommendation': recommendation})

def recommendations(property_id):
    try:
        df = pd.read_csv("recommendation.csv")
        property_info = df.loc[
            df["propertyid"] == property_id, ["propertyid", "type", "address", "bedroom", "bathroom", "area ( Sq. Yd.)",
                                              "price", "purpose"]]
        property_detail = property_info.to_dict("records")[0]

        property_id = property_detail.get('propertyid')
        property_type = property_detail.get('type')
        property_address = property_detail.get('address')
        property_purpose = property_detail.get('purpose')
        property_area = property_detail.get('area ( Sq. Yd.)')
        property_bedroom = property_detail.get('bedroom')
        property_bathroom = property_detail.get('bathroom')
        property_price = property_detail.get('price')

        prediction = []
        prediction.append(property_id)
        prediction.append(property_address)
        prediction.append(property_type)
        prediction.append(property_purpose)
        prediction.append(property_bedroom)
        prediction.append(property_bathroom)
        prediction.append(property_area)
        prediction.append(property_price)

        y = df["image-src"]
        x = df.drop(columns=["location", "description", "mobile", "image-src"])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        x_train_df = x_train.copy()

        categorical_cols = ["address", "type", "purpose"]
        column_trans = make_column_transformer((OneHotEncoder(sparse=False, drop="first"),
                                                categorical_cols), (StandardScaler(),
                                                                    make_column_selector(dtype_include="number")))

        pipe_new = make_pipeline(column_trans)
        pipe_new.fit(x_train, y_train)
        x_train = pipe_new.fit_transform(x_train)

        kdt = KDTree(x_train, leaf_size=30, metric='euclidean')
        x_train_df.to_csv("X_Train.csv")

        testing = pd.DataFrame([prediction], columns=["propertyid", "address", "type", "purpose", "bedroom", "bathroom",
                                                      "area ( Sq. Yd.)", "price"])
        testing = pipe_new.transform(testing)
        indices = kdt.query(testing, k=5, return_distance=False)
        recommend = x_train_df.iloc[indices[0]]
        recommend = recommend["propertyid"].values
        return recommend
    except:
        all_property_ids_queryset = Properties.objects.values_list('Property_id', flat=True)

        all_property_ids = list(all_property_ids_queryset)

        recommend = random.sample(all_property_ids, 5)
        return recommend



def propertydetails(request,property_id):
    property = get_object_or_404(Properties, Property_id=property_id)
    map = inside_detail_map(property_id)
    recommend=recommendations(property_id)
    list=[]
    for i in recommend:
        list.append(i)
    recommendation = Properties.objects.filter(Property_id__in=list)
    return render(request, 'property_details.html', {'property': property, 'recommendation': recommendation,'map':map})



def future_price_prediction(property_id):
    data = pd.read_excel('Future Percentage.xlsx')
    df = pd.read_csv('PropertyData-Zameen.csv')
    property_info = df.loc[df['Propertyid'] == property_id, ['Property Address', 'Price']]
    property_detail = property_info.to_dict('records')[0]
    address = property_detail.get('Property Address')
    current_price = property_detail.get('Price')

    address_data = data[data['Address'] == address]
    if address_data.empty:
        address_data = data[data['Address'] == "Other"]

    current_month = pd.to_datetime(address_data['Date'].iloc[0], format='%Y-%m-%d').month
    current_year = pd.to_datetime(address_data['Date'].iloc[0], format='%Y-%m-%d').year

    address_data = address_data[
        pd.to_datetime(address_data['Date'], format='%Y-%m-%d') >= pd.to_datetime(f'{current_month}-{current_year}',
                                                                                  format='%m-%Y')]

    final_percentage = 1.0
    for i, row in address_data.iterrows():
        percentage_change = 1 + (row['Percentage_Change'] / 100)
        final_percentage *= percentage_change
    future_price = current_price * final_percentage
    return round(future_price)
    # df = pd.read_excel('rentaldata.xlsx', na_values=[''])
    # df = df.dropna(axis=1, how='all')
    # df = df[['propertyid','location', 'type', 'bedroom', 'bathroom', 'area ( Sq. Yd.)', 'purpose', 'price']]
    # df = df.fillna(0)

    # define a function to convert the price string to a float
    # def convert_price(price):
    #     if isinstance(price, str):
    #         price = price.replace(',', '').lower()
    #         if 'lakh' in price:
    #             return float(price.replace('lakh', '')) * 100000
    #         elif 'thousand' in price:
    #             return float(price.replace('thousand', '')) * 1000
    #         elif 'crore' in price:
    #             return float(price.replace('crore', '')) * 10000000
    #         else:
    #             return float(price)
    #     else:
    #         return price

    # data = pd.read_excel('percentagedata.xlsx')
    # address_data = data[data['Address'] == address]
    # current_month = pd.to_datetime(address_data['Date'].iloc[0], format='%b-%y').month
    # current_year = pd.to_datetime(address_data['Date'].iloc[0], format='%b-%y').year
    #
    # address_data = address_data[
    #     pd.to_datetime(address_data['Date'], format='%b-%y') >= pd.to_datetime(f'{current_month}-{current_year}',
    #                                                                            format='%m-%Y')]
    #
    # final_percentage = 1.0
    # for i, row in address_data.iterrows():
    #     percentage_change = 1 + (row['Percentage_Change'] / 100)
    #     final_percentage *= percentage_change
    # future_price = current_price * final_percentage
    # return round(future_price)
    #
    # df_new = df.drop(['price'], axis=1)
    # desired_propertyid = property_id
    # propertydetail_arr = np.array(df_new.loc[df_new['propertyid'] == desired_propertyid]).reshape(1, -1)
    # price = df.loc[df['propertyid'] == desired_propertyid, 'price'].values[0]
    # prediction = forest.predict(propertydetail_arr)
    # number = float(prediction[0])
    # predictvalue = round(number)
    # difference =predictvalue - price
    # locale.setlocale(locale.LC_ALL, 'ur_PK')
    # val = locale.format_string('%.0f', difference, grouping=True)
    # val2 = locale.format_string('%.0f', predictvalue, grouping=True)
    # list = []
    # list2=[]
    # if difference > 0:
    #
    #     list = [val2,"increased",val]
    #     list2 = [price,difference]
    # elif difference < 0:
    #     dif = abs(int(difference))
    #     val2 = locale.format_string('%.0f', dif, grouping=True)
    #     list = [val2,"decreased",val2]
    #     list2 = [price, difference]
    # else:
    #     list = [val2,"none", val]
    #     list2 = [price, difference]
    # return list,list2


def futureprices(request, property_id):
    property = get_object_or_404(Properties, Property_id=property_id)
    predictionvalue = future_price_prediction(property_id)
    # increase_percentage = round(((int(predictionvalue) - int(current_price)) / int(current_price)) * 100, 2)
    # print(increase_percentage)
    Price = property.price
    x = ['2023', '2024']
    y = [int(Price), int(predictionvalue)]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Set the font size of the axis labels
    plt.rcParams.update({'font.size': 12})

    ax.bar(x, y, width=0.3, color=['black'] + ['orange'] * (len(x) - 1))

    ax.set_xlabel('Future Price Prediction', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)

    # Fit the y-axis label inside the PNG
    plt.tight_layout()

    filename = 'futurebar_graph.png'
    path = os.path.join(os.getcwd(), 'PropertyGuru', 'static', 'img', filename)
    plt.savefig(path)
    return render(request, 'future_Prices.html',{'property': property, "predictionvalue": predictionvalue})


def currentworthmodel(property_id):
    warnings.simplefilter("ignore")

    df = pd.read_csv("Responses.csv")
    data = pd.read_csv('PropertyData-Zameen.csv')

    percentage_value = {}
    for column in df.columns:
        if df[column].dtype == "int64":
            mean_value = df[column].mean()
            column_int = int(column)
            percentage = round((mean_value / column_int) * 100, 2)
            percentage_value[column_int] = percentage

    main_nested_dict = {'For Rent': {}, 'For Sale': {}}
    for key in list(percentage_value.keys())[:9]:
        main_nested_dict['For Rent'][key] = percentage_value[key]
    for key in list(percentage_value.keys())[9:]:
        main_nested_dict['For Sale'][key] = 95

    def get_property_info(property_id):
        property_info = data.loc[data['Propertyid'] == property_id, ['Purpose', 'Price']]
        return property_info.to_dict('records')[0]

    def calculate_discount(main_nested_dict, get_property_info):
        purpose = get_property_info['Purpose']
        price = get_property_info['Price']
        discount_dict = main_nested_dict.get(purpose)
        if discount_dict is None:
            raise ValueError(f"No discount information found for '{purpose}' purpose.")
        for key in sorted(discount_dict.keys(), reverse=True):
            if price >= key:
                discount = discount_dict[key]
                discounted_value = (discount / 100) * price
                return int(discounted_value)
        return price

    def encode_address(address):
        with open('currentworth.pkl', 'rb') as file:
            saved_data = pickle.load(file)
            label_encoder = saved_data['label_encoder']
        return label_encoder.transform([address])[0]

    def currentworth(property_id):
        with open('currentworth.pkl', 'rb') as file:
            saved_data = pickle.load(file)
            model = saved_data['model']

        property_info = data.loc[
            data['Propertyid'] == property_id, ['Property Address', 'Type', 'Purpose', 'Area (in Sq. Yd.)', 'Bedroom',
                                                'Bathroom', 'Price']]
        property_detail = property_info.to_dict('records')[0]

        prediction = []
        zero = 0
        one = 1
        property_price = property_detail.get('Price')
        print('Property Price on Website =', property_price)
        property_address = property_detail.get('Property Address')
        propertyadd_id = encode_address(property_address)
        property_type = property_detail.get('Type')
        property_purpose = property_detail.get('Purpose')
        property_area = property_detail.get('Area (in Sq. Yd.)')
        property_bedroom = property_detail.get('Bedroom')
        property_bathroom = property_detail.get('Bathroom')

        prediction.append(int(property_area))
        prediction.append(int(property_bedroom))
        prediction.append(int(property_bathroom))
        if property_type == "House":
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(one)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
        elif property_type == "Flat":
            prediction.append(zero)
            prediction.append(one)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
        elif property_type == "Farm House":
            prediction.append(one)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
        elif property_type == "Lower Portion":
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(one)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
        elif property_type == "Penthouse":
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(one)
            prediction.append(zero)
            prediction.append(zero)
        elif property_type == "Room":
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(one)
            prediction.append(zero)
        elif property_type == "Upper Portion":
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(zero)
            prediction.append(one)
        else:
            print("Property Type {} Not In The List!", property_type)
        if property_purpose == "For Sale":
            prediction.append(zero)
            prediction.append(one)
        elif property_purpose == "For Rent":
            prediction.append(one)
            prediction.append(zero)
        else:
            print("Property Purpose {} Not In The List!", property_purpose)
        prediction.append(propertyadd_id)

        propertydetail_arr = np.array(prediction).reshape(1, -1)
        prediction = model.predict(propertydetail_arr)
        minimum_value = round(float(prediction[0]))

        threshold_value = 0.8 * property_price
        if minimum_value < property_price:
            if minimum_value < threshold_value:
                minimum_value = threshold_value
        else:
            minimum_value = threshold_value

        maximum_value = calculate_discount(main_nested_dict, get_property_info(property_id))
        maximum_value = round(maximum_value)

        if minimum_value < maximum_value:
            print("The minimum value of the property id {} is : {} , and the maximum value is {}.".format(property_id,
                                                                                                          minimum_value,
                                                                                                          maximum_value))
            list1=[minimum_value,maximum_value]
            return list1

        elif minimum_value == maximum_value:
            print("The minimum value & maximum value of the property id {} is : {}.".format(property_id, minimum_value))
            return [minimum_value,maximum_value]

        elif minimum_value > maximum_value:
            print("The minimum value & maximum value of the property id {} is : {}.".format(property_id, maximum_value))
            return [minimum_value,maximum_value]

    maxmin = currentworth(property_id)
    # desired_propertyid = property_id
    # propertydetail_arr = np.array(df_new.loc[df_new['propertyid'] == desired_propertyid]).reshape(1, -1)
    # price = df.loc[df['propertyid'] == desired_propertyid, 'price'].values[0]
    # prediction = forest.predict(propertydetail_arr)
    # number = float(prediction[0])
    # minimumval = [round(number)]
    return maxmin

def currentworth(request,property_id):
    property = get_object_or_404(Properties, Property_id=property_id)
    value = currentworthmodel(property_id)
    # predictionvalue = future_price_prediction(property_id)
    x = ['Min', 'Max']
    y = [int(value[0]), int(value[1])]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Set the font size of the axis labels
    plt.rcParams.update({'font.size': 12})

    ax.bar(x, y, width=0.3, color=['black'] + ['orange'] * (len(x) - 1))

    ax.set_xlabel('Current Worth', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)

    # Fit the y-axis label inside the PNG
    plt.tight_layout()

    filename = 'worthbar_graph.png'
    path = os.path.join(os.getcwd(), 'PropertyGuru', 'static', 'img', filename)
    plt.savefig(path)
    return render(request,'current_worth.html', {'property': property,'value':value})

def inside_detail_map(property_id):
    data = Properties.objects.filter(Property_id=property_id).values('Property_id', 'location', 'area_sq', 'purpose',
                                                                     'property_type', 'price', 'image_src')
    geocoded_data = []
    for row in data:
        location = row['location'],
        url = 'https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyB12RPoQsTv-ZGF7DeGiaqoZ7w-OSu7RYs&address=%s' % location
        response = requests.get(url)
        if response.status_code == 200:
            result = json.loads(response.content)
            if result['status'] == 'OK':
                lat = result['results'][0]['geometry']['location']['lat']
                lng = result['results'][0]['geometry']['location']['lng']
                geocoded_data.append(
                    {'Property_id': row['Property_id'], 'location': location, 'lat': lat, 'lng': lng,
                     'area_sq': row['area_sq'], 'purpose': row['purpose'],
                     'property_type': row['property_type'],
                     'price': row['price'], 'image_src': row['image_src']
                     })

    # Initialize a list to hold the features
    features = []

    # Iterate over the data and create a feature for each record
    for row in geocoded_data:
        feature = {
            'type': 'Feature',
            'properties': {
                'Property_id': row['Property_id'],
                'location': row['location'],
                'area_sq': row['area_sq'],
                'price': row['price'],
                'purpose': row['purpose'],
                'property_type': row['property_type'],
                'image_src': row['image_src']
            },
            'geometry': {
                'type': 'Point',
                'coordinates': [row['lng'], row['lat']]
            }
        }
        features.append(feature)
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    # Convert the GeoJSON object to a JSON string
    json_data = json.dumps(geojson)

    return json_data

def view_on_map(request,property_id):
    property = get_object_or_404(Properties, Property_id=property_id)
    data = Properties.objects.filter(Property_id=property_id).values('Property_id','location', 'area_sq', 'purpose', 'property_type', 'price', 'image_src')
    geocoded_data = []
    for row in data:
        location = row['location'],
        url = 'https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyB12RPoQsTv-ZGF7DeGiaqoZ7w-OSu7RYs&address=%s' % location
        response = requests.get(url)
        if response.status_code == 200:
            result = json.loads(response.content)
            if result['status'] == 'OK':
                lat = result['results'][0]['geometry']['location']['lat']
                lng = result['results'][0]['geometry']['location']['lng']
                geocoded_data.append(
                    {'Property_id':row['Property_id'],'location': location, 'lat': lat, 'lng': lng, 'area_sq': row['area_sq'], 'purpose': row['purpose'],
                     'property_type': row['property_type'],
                     'price': row['price'], 'image_src': row['image_src']
                     })

    # Initialize a list to hold the features
    features = []

    # Iterate over the data and create a feature for each record
    for row in geocoded_data:
        feature = {
            'type': 'Feature',
            'properties': {
                'Property_id': row['Property_id'],
                'location': row['location'],
                'area_sq': row['area_sq'],
                'price': row['price'],
                'purpose': row['purpose'],
                'property_type': row['property_type'],
                'image_src': row['image_src']
            },
            'geometry': {
                'type': 'Point',
                'coordinates': [row['lng'], row['lat']]
            }
        }
        features.append(feature)
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    # Convert the GeoJSON object to a JSON string
    json_data = json.dumps(geojson)
    return render(request, 'viewonmap.html', {'property': property,'json_data': json_data})
#
#

def nearbyproperties(request):
    try:
        if request.method == "POST":
            areaname = request.POST.get('areaname')
        data = Properties.objects.filter(address__icontains=areaname).values('Property_id', 'location', 'area_sq',
                                                                             'purpose', 'property_type', 'price',
                                                                             'image_src')

        geocoded_data = []
        for row in data:
            location = row['location'],
            url = 'https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyB12RPoQsTv-ZGF7DeGiaqoZ7w-OSu7RYs&address=%s' % location
            response = requests.get(url)
            if response.status_code == 200:
                result = json.loads(response.content)
                if result['status'] == 'OK':
                    lat = result['results'][0]['geometry']['location']['lat']
                    lng = result['results'][0]['geometry']['location']['lng']
                    geocoded_data.append(
                        {'property_id': row['Property_id'], 'location': location, 'lat': lat, 'lng': lng,
                         'area_sq': row['area_sq'], 'purpose': row['purpose'],
                         'property_type': row['property_type'],
                         'price': row['price'], 'image_src': row['image_src']
                         })
        features = []
        # Iterate over the data and create a feature for each record
        for row in geocoded_data:
            feature = {
                'type': 'Feature',
                'properties': {
                    'property_id': row['property_id'],
                    'location': row['location'],
                    'area_sq': row['area_sq'],
                    'price': row['price'],
                    'purpose': row['purpose'],
                    'property_type': row['property_type'],
                    'image_src': row['image_src']
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['lng'], row['lat']]
                }
            }
            features.append(feature)
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }

        # Convert the GeoJSON object to a JSON string
        json_data = json.dumps(geojson)

        return render(request, 'nearbyproperties.html', {'json_data': json_data})
    except:
        return render(request,'Error.html');

def nearbyproperties_rent(request):
    if request.method == "POST":
        areaname = request.POST.get('areaname')

    data = Properties.objects.filter(address__icontains=areaname, purpose='For Rent').values('Property_id', 'location',
                                                                                             'area_sq', 'purpose',
                                                                                             'property_type', 'price',
                                                                                             'image_src')
    geocoded_data = []
    for row in data:
        location = row['location'],
        url = 'https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyB12RPoQsTv-ZGF7DeGiaqoZ7w-OSu7RYs&address=%s' % location
        response = requests.get(url)
        if response.status_code == 200:
            result = json.loads(response.content)
            if result['status'] == 'OK':
                lat = result['results'][0]['geometry']['location']['lat']
                lng = result['results'][0]['geometry']['location']['lng']
                geocoded_data.append(
                    {'property_id': row['Property_id'], 'location': location, 'lat': lat, 'lng': lng,
                     'area_sq': row['area_sq'], 'purpose': row['purpose'],
                     'property_type': row['property_type'],
                     'price': row['price'], 'image_src': row['image_src']
                     })
    features = []
    # Iterate over the data and create a feature for each record
    for row in geocoded_data:
        feature = {
            'type': 'Feature',
            'properties': {
                'property_id': row['property_id'],
                'location': row['location'],
                'area_sq': row['area_sq'],
                'price': row['price'],
                'purpose': row['purpose'],
                'property_type': row['property_type'],
                'image_src': row['image_src']
            },
            'geometry': {
                'type': 'Point',
                'coordinates': [row['lng'], row['lat']]
            }
        }
        features.append(feature)
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    # Convert the GeoJSON object to a JSON string
    json_data = json.dumps(geojson)

    return render(request, 'nearbyproperties.html', {'json_data': json_data})


def nearbyproperties_purchase(request):
    if request.method == "POST":
        areaname = request.POST.get('areaname')
    data = Properties.objects.filter(address__icontains=areaname, purpose='For Sale').values('Property_id', 'location',
                                                                                             'area_sq', 'purpose',
                                                                                             'property_type', 'price',
                                                                                             'image_src')
    geocoded_data = []
    for row in data:
        location = row['location'],
        url = 'https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyB12RPoQsTv-ZGF7DeGiaqoZ7w-OSu7RYs&address=%s' % location
        response = requests.get(url)
        if response.status_code == 200:
            result = json.loads(response.content)
            if result['status'] == 'OK':
                lat = result['results'][0]['geometry']['location']['lat']
                lng = result['results'][0]['geometry']['location']['lng']
                geocoded_data.append(
                    {'property_id': row['Property_id'], 'location': location, 'lat': lat, 'lng': lng,
                     'area_sq': row['area_sq'], 'purpose': row['purpose'],
                     'property_type': row['property_type'],
                     'price': row['price'], 'image_src': row['image_src']
                     })
    features = []
    # Iterate over the data and create a feature for each record
    for row in geocoded_data:
        feature = {
            'type': 'Feature',
            'properties': {
                'property_id': row['property_id'],
                'location': row['location'],
                'area_sq': row['area_sq'],
                'price': row['price'],
                'purpose': row['purpose'],
                'property_type': row['property_type'],
                'image_src': row['image_src']
            },
            'geometry': {
                'type': 'Point',
                'coordinates': [row['lng'], row['lat']]
            }
        }
        features.append(feature)
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    # Convert the GeoJSON object to a JSON string
    json_data = json.dumps(geojson)

    return render(request, 'nearbyproperties.html', {'json_data': json_data})

