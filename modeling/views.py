from __future__ import unicode_literals
from django.shortcuts import render
import os
from django.http import JsonResponse
from sklearn.externals import joblib

CURRENT_DIR = os.path.dirname(__file__)
model_file = os.path.join(CURRENT_DIR, 'model.file')
model = joblib.load(model_file)

# Create your views here.


def api_sentiment_pred(request):
    review = request.GET['review']
    result = 'Positive' if model.predict([review]) else 'Negative'
    # https://suwoni-codelab.com/django/2018/03/25/Django-FBV/ 여기 글 참조할것.JsonResponse
    return JsonResponse(result, safe=False)
