from django.contrib import admin
from django.shortcuts import render
from django.contrib import messages
from .models import Properties,Current_worth,Future_price,Feedback
from django.urls import path, reverse
from django import forms
from django.http import HttpResponseRedirect
from io import BytesIO
import pandas as pd


class CsvImportForm(forms.Form):
    csv_upload = forms.FileField()
class PropertiesAdmin(admin.ModelAdmin):
    list_display = ('Property_id','location','property_type','address','bedrooms','bathrooms','area_sq','purpose','price','property_description','mobile','image_src','image_src2','image_src3','image_src4')

    def get_urls(self):
        urls = super().get_urls()
        new_urls = [path('upload-csv/',self.upload_csv),]
        return new_urls + urls

    def upload_csv(self, request):
        if request.method == 'POST':
            csv_file = request.FILES["csv_upload"]
            if not csv_file.name.endswith('.xlsx'):
                messages.warning(request, 'The wrong file type was uploaded')
                return HttpResponseRedirect(request.path_info)
            file_data = csv_file.read()
            xls_data = pd.read_excel(BytesIO(file_data))
            for index, row in xls_data.iterrows():
                created = Properties.objects.update_or_create(
                    Property_id=row[0],
                    location=row[1],
                    property_type=row[2],
                    address=row[3],
                    bedrooms=row[4],
                    bathrooms=row[5],
                    area_sq=row[6],
                    purpose=row[7],
                    price=row[8],
                    property_description=row[9],
                    mobile=row[10],
                    image_src=row[11],
                    image_src2=row[12],
                    image_src3=row[13],
                    image_src4=row[14],

                )
            url = reverse('admin:index')
            return HttpResponseRedirect(url)
        form = CsvImportForm()
        data = {"form": form}
        return render(request, "admin/csv_upload.html", data)

admin.site.register(Properties,PropertiesAdmin),
admin.site.register(Current_worth),
admin.site.register(Future_price),
admin.site.register(Feedback),




