Repository for pf homework

```
pf-model
│   README.md - Setup guide
│
└───mlflowserver
│   │   file011.txt
│   │   file012.txt
│   │
│   └───subfolder1
│       │   file111.txt
│       │   file112.txt
│       │   ...
│   
└───folder2
    │   file021.txt
    │   file022.txt
```

# I. Architecture

Simple architecture
![simpleArchitecture](images/simple-architecture.png)

More details
![Architecture](images/architecture.png)

- Source Repository: A central repository. In this repo we can using both local or github

- MLFlow: Training components

    - Defines environment, parameters and model’s interface.
    - Track experiment results and hyperparameters
    - Snapshot / version of the model.
    - Model registry

- Feature / Artifacts database: Database of training data, artifacts of trained model. Local file in this example, its can be migrated to s3/ gcs...  later

- Image repositories: Docker hub (on cloud can using GCP Image repositories/ AWS Container Registry)

- Seldon Core: Serving / Inference services

  *"Easier and faster to deploy your machine learning models and experiments at scale on Kubernetes"*

    - Preprocessing request: There are 3 popular way to do
      - a
      - b
      - c

    - Dashboard

    - Endpoints

# II. Pre-requisites 🧰

## Anaconda

We're using Anaconda for easy manage develop environment

Details : https://docs.anaconda.com/anaconda/install/windows/

Create new environment
```bash
    !conda create --n podfoods python=3.8
```
## MLFlow

<!-- ![mlflow](images\MLflow-logo-final-white-TM.png) -->

```bash
    !pip install mlflow
```

## Seldon

<!-- ![seldoncore](images\Seldon_Logo_0.jpg) -->

Seldon core is recommended install on GCP Cloud environment

- Seldon Core

Setup document details: https://docs.seldon.io/projects/seldon-core/en/latest/nav/installation.html

Local Port Forwarding

```bash
    !kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
```

Dashboard (Grafana)

## s2i (Source to Image)

https://github.com/openshift/source-to-image#installation

We using s2i to build docker later


# III. Training model

```bash
    !python model\train.py 3 50 5
    or
    !mlflow run model -P max_depth=3 n_estimators=50 data_month=5
```

training avaiable in model\train_model.ipynb

# IV. Model exprimental management

```bash
    !mlflow ui
```
http://localhost:5000
![mlflowui](images/mlflow-ui.png)

custom port
```bash
    !mlflow ui ....
```
# V. Build model docker image

```bash
    !make
```

Test docker image locall

```bash
    !docker run -it --rm -p 8080:8080 -e PREDICTIVE_UNIT_PARAMETERS='[{"type":"STRING","name":"model_uri","value":"file:///model"}]' -v /home/haunv_it/mlflow_wine_artifact/elasticnet_wine:/model haunv/mlflowservercustom:1.16.0-dev
```

Test request/respone

```bash
    !curl ...
```

Push to docker hub

Note: Edit corresponding to your docker hub repositories

```bash
    !docker push haunv/mlflowservercustom:lastest
```

# VI. Deploy model docker image 

Edit seldon docker servers values (MLFLOW_SERVER_CUSTOM)

```yaml
predictor_servers:
...
  TEMPO_SERVER:
    protocols:
      kfserving:
        defaultImageVersion: "0.3.2"
        image: seldonio/mlserver
  MLFLOW_SERVER_CUSTOM:
    protocols:
      seldon:
        defaultImageVersion: "1.15.0-dev"
        image: haunv/mlflowservercustom
    
```

Register custom server docker with seldon core

```bash
!git clone seldon-core

!helm upgrade seldon-core seldon-core/helm-charts/seldon-core-operator --namespace seldon-system --values values.yaml --set istio.enabled=true
    
```


deploy YAML file

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: mlflow
  namespace: seldon
spec:
  name: wines
  predictors:
  - componentSpecs:
    - spec:
        # We are setting high failureThreshold as installing conda dependencies
        # can take long time and we want to avoid k8s killing the container prematurely
        containers:
        - name: classifier
          livenessProbe:
            initialDelaySeconds: 80
            failureThreshold: 200
            periodSeconds: 5
            successThreshold: 1
            httpGet:
              path: /health/ping
              port: http
              scheme: HTTP
          readinessProbe:
            initialDelaySeconds: 80
            failureThreshold: 200
            periodSeconds: 5
            successThreshold: 1
            httpGet:
              path: /health/ping
              port: http
              scheme: HTTP
        imagePullPolicy: Always
    graph:
      children: []
      implementation: MLFLOW_SERVER_CUSTOM
    # Replace with corresponding artifact folder
      modelUri: gs://seldon-models/v1.10.0-dev/mlflow/elasticnet_wine
      name: classifier
    name: default
    replicas: 1
```

Deploy

```bash
!kubectl apply -f filename.yaml
```

Request

```bash
!curl ....
```

# VII. Dash board
