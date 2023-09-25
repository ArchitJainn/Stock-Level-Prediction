"""
URL configuration for inventory_management project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from inventory_forecasting import views

urlpatterns = [
    path('about', views.index, name='about'),
    path('fun', views.func, name='fun'),
    path('forecasting', views.inventory_forecasting, name='inventory'),
    path('addItemStock', views.addItemStock, name='addItemStock'),
    path('orderItemStock', views.orderItemStock, name='orderItemStock')
]
