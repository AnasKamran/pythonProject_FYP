# Generated by Django 4.2 on 2023-04-15 15:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PropertyGuru', '0005_properties_address'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='properties',
            name='amenities',
        ),
        migrations.RemoveField(
            model_name='properties',
            name='propertylink_href',
        ),
        migrations.RemoveField(
            model_name='properties',
            name='title',
        ),
        migrations.AddField(
            model_name='properties',
            name='image_src2',
            field=models.CharField(blank=True, max_length=400, null=True),
        ),
        migrations.AddField(
            model_name='properties',
            name='image_src3',
            field=models.CharField(blank=True, max_length=400, null=True),
        ),
        migrations.AddField(
            model_name='properties',
            name='image_src4',
            field=models.CharField(blank=True, max_length=400, null=True),
        ),
        migrations.AlterField(
            model_name='properties',
            name='address',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='properties',
            name='mobile',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
        migrations.AlterField(
            model_name='properties',
            name='price',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='properties',
            name='property_description',
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
    ]