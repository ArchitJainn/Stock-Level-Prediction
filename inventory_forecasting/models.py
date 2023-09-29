from django.db import models

# Create your models here.

class Inventory_Item(models.Model):
    item_name = models.CharField(max_length=122)
    item_id = models.CharField(max_length=122, default=0, primary_key=True)
    item_quantity = models.IntegerField()
    date = models.DateField()
