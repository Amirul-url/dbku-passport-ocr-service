from django.urls import path, include

urlpatterns = [
    path("api/ocr/", include("passport_ocr.urls")),
]