import base64
import json
from io import BytesIO

import requests
from PIL import Image


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


img = Image.open("/data/yunshilin/yolo26-detect/fire_detect/dataset/test/images/WEB10625.jpg")
base64_str = image_to_base64(img)
json_str = json.dumps({"base64_strs": base64_str}, ensure_ascii=False)
payload = {
    "inputs": [
        {
            "name": "IMAGE",
            "shape": [1],
            "datatype": "BYTES",
            "data": [json_str]
        }
    ],
    "outputs": [
        {
            "name": "RESULT"
        }
    ]
}

response = requests.post(
    "http://localhost:19001/v2/models/fire_detect/infer",
    json=payload,
    timeout=30,
)

response.raise_for_status()
print(response.json())