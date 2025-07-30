import requests
import json
import numpy as np  
from img import img_base
ENDPOINT_URL = "http://localhost:8080/serve/car-detector"
# AUTH_TOKEN = "your-secret-bearer-token"  


payload = {
    "image_base64": img_base
}

 

print("Отправка запроса на Model Endpoint...")

try:
    response = requests.post(url=ENDPOINT_URL, json=payload)

    response.raise_for_status() 

    print("✅ Запрос выполнен успешно!")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))

except requests.exceptions.RequestException as e:
    print(f"❌ Произошла ошибка: {e}")
    if e.response:
        print(f"Ответ сервера: {e.response.text}")