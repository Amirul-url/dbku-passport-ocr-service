from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse
from django.urls import include, path, re_path
from django.views.static import serve


def health(request):
    return JsonResponse({
        "status": "ok",
        "service": "passport-ocr",
    })


urlpatterns = [
    path("", health),
    path("api/ocr/", include("passport_ocr.urls")),
]

# Development/local
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Production fallback for media files
if not settings.DEBUG:
    urlpatterns += [
        re_path(
            r"^media/(?P<path>.*)$",
            serve,
            {"document_root": settings.MEDIA_ROOT},
        ),
    ]