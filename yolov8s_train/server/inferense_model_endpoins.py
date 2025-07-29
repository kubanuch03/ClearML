import requests
import json
import numpy as np  

ENDPOINT_URL = "https://serving.app.clear.ml/your-unique-endpoint-path"
AUTH_TOKEN = "your-secret-bearer-token"  

input_features = np.random.rand(1, 10).tolist() 

payload = {
    "inputs": [
        {
            "name": "input-0",   
            "shape": [1, 10],    
            "datatype": "FP32",  
            "data": input_features
        }
    ]
}

headers = {
    'Authorization': f'Bearer {AUTH_TOKEN}',
    'Content-Type': 'application/json'
}

print("Отправка запроса на Model Endpoint...")

try:
    response = requests.post(url=ENDPOINT_URL, headers=headers, data=json.dumps(payload))

    response.raise_for_status() 

    result_json = response.json()
    
    prediction = result_json['outputs'][0]['data']

    print("✅ Запрос выполнен успешно!")
    print(f"Предсказание модели: {prediction}")

except requests.exceptions.RequestException as e:
    print(f"❌ Произошла ошибка: {e}")
    if e.response:
        print(f"Ответ сервера: {e.response.text}")