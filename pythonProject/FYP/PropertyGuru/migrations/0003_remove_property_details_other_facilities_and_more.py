# Generated by Django 4.1.7 on 2023-04-02 08:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PropertyGuru', '0002_alter_current_worth_max_amount_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='property_details',
            name='other_facilities',
        ),
        migrations.RemoveField(
            model_name='property_details',
            name='rooms',
        ),
        migrations.RemoveField(
            model_name='rental_properties',
            name='area',
        ),
        migrations.RemoveField(
            model_name='selling_properties',
            name='area_name',
        ),
        migrations.AddField(
            model_name='property_details',
            name='amenities',
            field=models.CharField(blank=True, max_length=300, null=True),
        ),
        migrations.AddField(
            model_name='property_details',
            name='bedrooms',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='property_details',
            name='image_src',
            field=models.CharField(blank=True, max_length=400, null=True),
        ),
        migrations.AddField(
            model_name='property_details',
            name='title',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='rental_properties',
            name='propertylink_href',
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
        migrations.AddField(
            model_name='selling_properties',
            name='propertylink_href',
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
        migrations.AlterField(
            model_name='property_details',
            name='purpose',
            field=models.CharField(blank=True, max_length=30, null=True),
        ),
    ]