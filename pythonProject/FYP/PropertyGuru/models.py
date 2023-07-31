from django.db import models
from io import BytesIO

#<-----------------Properties---------------->
class Properties(models.Model):
    objects = [None]
    Property_id = models.IntegerField(primary_key=True,default=0)
    location = models.CharField(max_length=100,blank=True,null=True)
    property_type = models.CharField(max_length=10, blank=True, null=True)
    address = models.CharField(max_length=50, blank=True, null=True)
    bedrooms = models.IntegerField(blank=True, null=True)
    bathrooms = models.IntegerField(blank=True,null=True)
    area_sq = models.CharField(max_length=10,blank=True,null=True)
    purpose = models.CharField(max_length=30, blank=True, null=True)
    price = models.CharField(max_length=100,blank=True,null=True)
    property_description = models.CharField(max_length=500,blank=True,null=True)
    mobile = models.CharField(max_length=20,blank=True, null=True)
    image_src = models.CharField(max_length=400, blank=True, null=True)
    image_src2 = models.CharField(max_length=400, blank=True, null=True)
    image_src3= models.CharField(max_length=400, blank=True, null=True)
    image_src4 = models.CharField(max_length=400, blank=True, null=True)



#<-----------------Predictions---------------->
class Current_worth(models.Model):
    p_id = models.ForeignKey(Properties, on_delete=models.CASCADE, default=0)
    max_amount = models.IntegerField()
    min_amount = models.IntegerField()

class Future_price(models.Model):
    p_id = models.ForeignKey(Properties, on_delete=models.CASCADE,default=0)
    predicted_amount= models.IntegerField()
    increase_by = models.IntegerField()
    decrease_by = models.IntegerField()
#<-----------------Recommendations---------------->

#------------------------Contact Us------------------------
class Feedback(models.Model):
    name = models.CharField(max_length=100)
    E_mail = models.CharField(max_length=100)
    ph_no = models.IntegerField()
    subject = models.CharField(max_length=100)
    Details = models.CharField(max_length=400)

    def __str__(self):
        return self.name
