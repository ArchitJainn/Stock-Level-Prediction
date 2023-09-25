from django.db import models

# Create your models here.

class Inventory_Item(models.Model):
    item_name = models.CharField(max_length=122)
    item_quantity = models.IntegerField(max_length=10)
    date = models.DateField()
