from pinferencia import Server, task
from PIL import Image
from io import BytesIO
import base64
import requests
import layoutparser as lp
import cv2

url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
image = Image.open(requests.get(url, stream=True).raw)

buffered = BytesIO()
image.save(buffered, format="JPEG")
encoded_string = base64.b64encode(buffered.getvalue())


response = requests.post(
    url="http://localhost:8000/v1/models/layout/predict",
    json={
        "data": encoded_string.decode("utf-8")
    },
)

print("Prediction:", response.json()["data"])
