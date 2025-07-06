import requests
from io import BytesIO
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
# تحميل النموذج
model = load_model('keras_model.h5', compile=False)
# تحميل labels
class_names = open("labels.txt", "r").readlines()
# تحميل الصورة من الانترنت
url = 'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg'
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
# تجهيز الصورة
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array
# التنبؤ
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]
# طباعة النتيجة
print(f"Class: {class_name[2:].strip()}")
print(f"Confidence Score: {confidence_score * 100:.2f}%")
