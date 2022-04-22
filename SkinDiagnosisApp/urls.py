from django.urls import path
from . import views

urlpatterns=[
    path('',views.mainpage,name="startpage"),
    path('about_page',views.about_page,name="About")
]

