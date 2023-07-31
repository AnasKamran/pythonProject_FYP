from django.contrib import admin
from . import views
from django.urls import path
from .views import contact


urlpatterns = [
    path('', views.index, name='index'),
    path('admin/', admin.site.urls),
    path('forms/',views.contact, name='contact'),
    path('about/', views.about, name='about'),
    path('availableproperties/',views.search,name='availableproperties'),
    path('contact/',views.contact,name='çontact'),
    path('nearbyproperties',views.nearbyproperties,name='nearbyproperties'),
    path('propertydetails/<int:property_id>/', views.propertydetails, name='propertydetails'),
    path('futureprices/<int:property_id>/', views.futureprices, name='futureprice'),
    path('currentworth/<int:property_id>/', views.currentworth, name='currentworth'),
    path('nearbyproperties-purchase/', views.nearbyproperties_purchase, name='nearbyproperties_purchase'),
    path('nearbyproperties-rent/', views.nearbyproperties_rent, name='nearbyproperties_rent'),
    path('viewonmap/<int:property_id>/', views.view_on_map, name='viewonmap'),
    path('Error/', views.Errors, name='Error')
]


