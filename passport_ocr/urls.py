from django.urls import path
from .views import extract_passport

urlpatterns = [
    path("passport/extract/", extract_passport),
]