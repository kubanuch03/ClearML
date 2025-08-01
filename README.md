```bash
clearml-init
```

```bash
clearml-serving create --name "Detect Serving Service"
```
<!-- clearml-serving create --name "My Local Serving Service" -->

```bash
clearml-serving --id "ba5efca98e684572a9a740d919e0b3e8" model add --engine custom --endpoint "car-detector" --project "Car Detection Project" --name "yolov8s_cars_finetuned3" --preprocess "../preprocess_prod.py"
```
from typing import Any

import numpy as np


# Notice Preprocess class Must be named "Preprocess"



```bash
cd clearml-serving/docker
docker compose up --build -d
```


```bash
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
```

 






 