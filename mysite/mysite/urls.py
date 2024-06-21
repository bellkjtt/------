from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('yolov8/', include('yolov8app.urls')),
]
