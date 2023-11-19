from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse,HttpResponseBadRequest
from PIL import Image
from io import BytesIO
from base64 import b64decode
import base64
import numpy as np
import os
from tensorflow.lite.python.interpreter import Interpreter

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from django.core.files import File
from django.contrib.auth.models import User
from django.contrib import messages
from rest_framework import status
from rest_framework.response import Response
from django.http import JsonResponse

# Create your views here.
# Create your views here.
arabic_letters = {
    'ء': {46: 'ء'},
    'أ': {91: 'أ',40: 'ـأ'},
    'ؤ': {41: 'ؤ'},
    'إ': {19: 'إ',90: 'ـإ'},
    'ا': {105: 'ا',39: 'ـا'},
    'ئ': {37: 'ـئ',  74: 'ـئـ'},
    'ب': {16: 'ب', 31: 'ـب', 100: 'بـ', 4: 'ـبـ'},
    'ت': {1: 'ت', 77: 'ـت', 61: 'تـ', 102: 'ـتـ'},
    'ث': {82: 'ث', 23: 'ـث', 2: 'ثـ', 58: 'ـثـ'},
    'ج': {99: 'ج', 27: 'ـج', 48: 'جـ', 89: 'ـجـ'},
    'ح': {9: 'ح', 54: 'ـح', 81: 'حـ', 17: 'ـحـ'},
    'خ': {12: 'خ', 73: 'ـخ', 76: 'خـ',79: 'ـخـ'},
    'د': {93: 'د', 84: 'ـد'},
    'ذ': {65: 'ذ', 94: 'ـذ'},
    'ر': {34: 'ر', 71: 'ـر'},
    'ز': {88: 'ز', 8: 'ـز'},
    'س': {63: 'س', 0: 'ـس', 49: 'سـ', 43: 'ـسـ'},
    'ش': {22: 'ش', 38: 'ـش', 62: 'شـ', 20: 'ـشـ'},
    'ص': {52: 'ص', 5: 'ـص', 80: 'صـ', 104: 'ـصـ'},
    'ض': {68: 'ض', 75: 'ـض', 29: 'ضـ', 69: 'ـضـ'},
    'ط': {28: 'lط', 87: 'ـط', 50: 'طـ', 98: 'ـط'}, #i am not sure
    'ظ': {45: 'ظ',26: 'ـظ', 70: 'ظـ', 15: 'ـظـ', 'medial2': '\uFECA'},#i am not sure
    'ع': {13: 'ع', 3: 'ـع', 32: 'عـ', 14: 'ـعـ'},
    'غ': {57: 'غ', 96: 'ـغ', 85: 'غـ', 21: 'gyn middle'},# iam not sure
    'ف': {107: 'ف', 25: 'ـف', 55: 'فـ', 51: 'ـفـ'},
    'ق': {47: 'ق', 101: 'ـق', 18: 'قـ', 97: 'ـقـ', 'medial2': '\uFEDE'},
    'ك': {44: 'ك', 103: 'ـك', 78: 'كـ', 66: 'ـكـ'},
    'ل': {106: 'ل', 7: 'ـل', 30: 'لـ', 56: 'ـلـ'},
    'م': {35: 'م', 11: 'ـم', 64: 'مـ', 95: 'ـمـ'},
    'ن': {72: 'ن', 86: 'ـن',24: 'نـ', 92: 'ـنـ'},
    'ه': {67: 'اه', 60: 'ـه', 6: 'هـ', 42: 'ـهـ'},
    'و': {33: 'و',36: 'ـو'},
    'ي': {59: 'ي', 10: 'ـي', 83: 'يـ', 53: 'ـيـ', 'medial2': '\uFEF6'},
 
}

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'DeepModel', 'model.tflite')


def get_outer_key(inner_key, my_dict):
    for outer_key, inner_dict in my_dict.items():
        if inner_key in inner_dict:
            value = inner_dict[inner_key]
            print(f"The value of '{inner_key}' is '{value}'")
            return outer_key
    return None




def convert_image(im):
            new_image = Image.new("RGBA", im.size, "WHITE")
            new_image.paste(im, (0, 0), im)
            if new_image.mode not in ("L", "RGB"):
                new_image = new_image.convert("RGB")
            new_image= new_image.resize((224,224))
            new_image=np.array(new_image)
            return new_image



 



def model_classification(img):
            interpreter =Interpreter(model_path)
            interpreter.allocate_tensors()
            input_data = np.expand_dims(img.astype(np.uint8), axis=0)
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_tensor = interpreter.get_tensor(output_details[0]['index'])
            output_array = output_tensor[0]
            predicted_class_index = np.argmax(output_array)
            print(predicted_class_index)
            inner_key = predicted_class_index
            outer_key = get_outer_key(inner_key, arabic_letters)
            print(f"The parent class of'{inner_key}' is '{outer_key}'")

            return inner_key,outer_key



 
def AutoCorrection(prediction_class,AcutalClass):

    if prediction_class!=AcutalClass:
        correction="خطأ حاول مرة اخرى"
    else:
        correction="احسنت اجابه صحيحة"
    print(correction)

    return correction

    




class ImageUploadLetterView(APIView):
    def post(self, request, format=None):
        text=request.data.get('letter')

        for key, image_file in request.FILES.items():
            # Process each uploaded image file
            # For example, you can save the file to a specific location or perform any required operations
            # Replace 'your_processing_logic' with your actual processing logic
            print(image_file)
            image=resize_image(image_file)
            image=convert_image(image)
            class_name,parent_class_name=model_classification(image)
        correctionOrnot=AutoCorrection(parent_class_name,text)
        print(correctionOrnot)

        return Response({'AutoWord': parent_class_name,'correction':correctionOrnot,'word':text})




class ImageUploadView(APIView):
    def post(self, request, format=None):
        word=''
        text=request.data.get('word')
        print(text,"-----")
        for key, image_file in request.FILES.items():
            # Process each uploaded image file
            # For example, you can save the file to a specific location or perform any required operations
            # Replace 'your_processing_logic' with your actual processing logic
            print(image_file)
            image=resize_image(image_file)
            image=convert_image(image)
            class_name,parent_class_name=model_classification(image)

            word+=parent_class_name
            print("word is append------",word)
            print(word)
        correctionOrnot=AutoCorrection(word[::-1],text)
        print(correctionOrnot)

        return Response({'AutoWord': word[::-1],'correction':correctionOrnot,'word':text})




def resize_image(image_file):
        
        
        # Open the image using PIL
        image = Image.open(image_file)

        # Resize the image
        resized_image = image.resize((224, 224))

        return resized_image