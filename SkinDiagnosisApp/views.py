from distutils.command.upload import upload
from django.shortcuts import render
from urllib import request
from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import os
from pathlib import Path
from django.core.files.storage import FileSystemStorage
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import csv
import geocoder
import folium
import requests
from Capstone_Project_AI.settings import BASE_DIR

def prediction(img_url):
    img_url='/Capstone_Project_AI'+img_url
    path=os.path.join(BASE_DIR,img_url)
    labels=['ACNE','ATOPIC DERMATITIS','BASAL CELL CARCINOMA','BENIGN KERATOSIS','CLEAR SKIN','ECZEMA',
       'MELANOCYSTIC NEVI','MELANOMA','PSORIASIS','SEBORRHEIC KERATOSES','FUNGAL INFECTION','WARTS']
    model=models.load_model(r'C:\Users\swati gour\capstone.h5')
    img=image.load_img(path,color_mode='rgb',target_size=(200,200,3))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val=model.predict(images)
    res=val.flatten()
    pred = np.where(res == np.amax(res))
    index=pred[0].tolist()
    print(labels[index[0]])
    return labels[index[0]]

def nearby_doctor():
    maps_api_key='dc_kutP1ys7SiKdEepmZuJiqS1Oii-UW_8cLuFAZvYw'
    g=geocoder.ip("me")
    myAddress=g.latlng
    url='https://discover.search.hereapi.com/v1/discover?at='+str(myAddress[0])+','+str(myAddress[1])+'&limit=5&q=hospital&in=countryCode:IND&apiKey='+maps_api_key
    data=requests.get(url).json()
    my_dict={}
    for i in range(5):
        if 'contacts' in data['items'][i].keys():
            contact_no=data['items'][i]['contacts'][0]['phone'][0]['value']
        my_dict[i]={'Hospital Name':data['items'][i]['title'],
        "Address":data['items'][i]['address']['label'],
        "Distance": str(data['items'][i]['distance'])+'metres',
        "contact_no":contact_no
        }
    return my_dict
# Create your views here.
def mainpage(request):
    if request.method=="POST":
        image_file=request.FILES['image']
        if image_file:
            fss=FileSystemStorage()
            file=fss.save('sample.jpg',image_file)
            file_url=fss.url(file)
            res=prediction(file_url)

            pred = res.lower()
            flag=0 
            string1 = []
            string2=""
            with open('D:\\Capstone_Project_AI\\templates\\file.csv', 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    if (row[0].lower()) == pred:
                        for i in row[1:-1]:
                            flag=1
                            string1.append(i)
                        string2 = row[-1]
                        print(string2)
            if flag==0:
                string1.append("No Data Avaialble")
            doc=nearby_doctor()

            return render(request,'startpage.html',{'prediction':res,'image_loc':file_url,'info':string1,'link':string2,'doc':doc})

    return render(request,'startpage.html',{'prediction':0})
def about_page(request):
    return render(request,'about_page.html')