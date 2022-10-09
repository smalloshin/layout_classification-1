from locust import HttpUser, task, between
from PIL import Image
from io import BytesIO
import base64


class PerformanceTests(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def testapi(self):
        image = Image.open('../samples/77.png')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_string = base64.b64encode(buffered.getvalue())
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        self.client.post("/v1/models/layout/predict", data=encoded_string.decode("utf-8"), headers=headers)
