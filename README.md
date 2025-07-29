```bash
clearml-init
```

```bash
clearml-serving create --name "My Local Serving Service"
```




```bash
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
```


```bash
clearml-serving --id 38b7e2eff3f74bf9adca1601810bc7be model add \
--engine custom \
--endpoint "car-detector" \
--preprocess "/home/zazaka/Dmain/ML/ClearML/yolov8s_train/serving_script_v2.py" \
--name "yolov8s_cars_finetuned3" \
--project "Car Detection Project"
```

```bash
curl -X POST "http://localhost:8080/serve/car-detector" \
-H "Content-Type: application/json" \
-d '{"image_url": "https://i.ytimg.com/vi/9mkXQYsRu90/maxresdefault.jpg"}'
```


# Развертывание ClearML Serving и ClearML Agent
```bash
pip install clearml-serving
clearml-serving
clearml-agent daemon --services-queue
```
```bash
pip install clearml-agent
```
## Config ClearML Serving
https://clear.ml/docs/latest/docs/clearml_serving/clearml_serving_setup#initial-setup

<!-- d3290336c62c7fb0bc8eb4046b60bc7f -->

<!-- 38b7e2eff3f74bf9adca1601810bc7be -->

```bash
clearml-serving create --name "My YOLOv8 Serving Service"
```
**Output**:<br>
    clearml-serving - CLI for launching ClearML serving engine
        New Serving Service created: id=b51b889c110c48c6b3732d09ce898244

#### copy id
 