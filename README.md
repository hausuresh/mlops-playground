Repository for pf homework

```
pf-model
│   
└───README.md - Guide
└───pf_toy_model_development.ipynb - development model jupyter notebook
└───mlflowserver
│   └───MLFlowServerCustom.py - Logic code for a new custom seldon core server with pre-processing data function
│   └───requirement.txt - required libraries for custom seldon core server
└───deploy - yaml files for deploy model to Seldon Core on Kubernetes  
└───images
└───mlruns - trained model artifacts
└───model
│   └───data - raw, traning & vocab data
│   └───train.py - MLFlow training model's code
│   └───train_model.ipynb - MLFlow training model's notebook
└───environment - define environment of custom Seldon server
└───requirement.txt - required libraries for training model
└───Makefile - Make custom seldon server docker
```

# I. Architecture

Simple architecture
![simpleArchitecture](images/simple-architecture.png)

More details
![Architecture](images/architecture.png)

- Source Repository: A central repository (local or github)

- MLFlow: Training components
    
    - Package ML model's code in a format to reproduce runs on any platform
    - Defines environment, parameters and model’s interface.
    - Track experiment results and hyperparameters
    - Snapshot / version of the model.
  
- Feature / Artifacts database: 
    - Training data, feature: local file in this example, its can be migrated to S3, GCS, Feature Store or other DB...  later. 
    - Artifacts database: artifacts of trained model.Note that in a production setting, MLflow would be configured to log models against a persistent data store (e.g. NFS, GCS, S3, Minio, etc.).

- Image repositories: Docker hub (or GCP Image repositories/ AWS Container Registry)

- Seldon Core: Serving / Inference services

  *"Easier and faster to deploy your machine learning models and experiments at scale on Kubernetes"*
  - Deploy model
  - Preprocessing request
  - Monitor: Dashboard of serving server's performance & model's metrics
  - And more functions can be added later
  
- Endpoints: endpoints post request & get result from Serving service by REST / GRPC

# II. Pre-requisites 🧰


```bash
!git clone https://github.com/hausuresh/pf-model.git
```

## Anaconda

We're using Anaconda for easy manage develop environment

Details : https://docs.anaconda.com/anaconda/install/

Create new environment
```bash
!conda create --n podfood python=3.7
```

## MLFlow

```bash
!pip install mlflow
```

## Seldon Core

Setup document on local / cloud. Details: https://docs.seldon.io/projects/seldon-core/en/latest/nav/installation.html

*Note: Seldon core is recommended install on GCP environment*

## S2I (Source to Image)

https://github.com/openshift/source-to-image#installation

s2i is used to build custom serving server later


# III. Training

0. Model development

```bash
!pip install -r requirement.txt
```

More details here ['pf_toy_model_development.ipynb'](pf_toy_model_development.ipynb)

1. MLflow Project

The MLproject file defines:
- Environment where training runs
- The parameters (alpha, l1_ratio in our case)

```YAML
# model\MLproject file
name: pf-toy-model

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.5}
      l1_ratio: {type: float, default: 0.1}
      data_month: {type: int, default: 5}
    command: "python train.py {alpha} {l1_ratio} {data_month}"
```

Train model command via MLFlow
```bash
!mlflow run model -P alpha=0.5 l1_ratio=0.1 data_month=5
```
or
```bash
!python model/train.py 0.5 0.1 5
```
or using central repositories 

```bash
!mlflow run https://github.com/pf-model...
```


Interactive with jupyter notebook model/train_model.ipynb

2. MLFlow Model tracking

MLFlow artifacts (parameters, evaluation metrics... ) can be stored on a remote server, which can then be shared across the entire team. However, on our example we will store these locally on a mlruns folder

```bash
!ls mlruns/0
```

Web UI

```bash
!mlflow ui
```
http://localhost:5000

![mlflowui](images/mlflow-ui.png)

Test serve model at local

```bash
!mlflow models serve -m mlruns/0/100ab95827c749d6803bb1093b36cd43/artifacts/model -p 1234

!curl --location --request POST 'http://127.0.0.1:1234/invocations' --header 'Content-Type: application/json' --data-raw '{ "data": [ [ 0, 0, 0, 0, 0, 28, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] ] }' 
```

Note: We do preprocessing later (at serving phase, so the input data for this model local test is in model's original features instead of store_id + product_id)


3. MLFlow Model

MLFlow artifacts are stored at mlruns folder

```bash
!ls mlruns/0
```

MLFlow python SDK allow us to get the best experiment id (more detail at model/train_model.ipynb)

```python
import mlflow
mlflow.search_runs(order_by=['metrics.RMSE ASC'], max_results=1)
```

Optional: Upload model artifacts

We will persist the models we have just trained using MLflow. For that, we will upload them into Google Cloud Storage.
Note that in a production setting, MLflow would be configured to log models against a persistent data store (e.g NFS, GCS, Minio, S3 etc.)

```bash
!gsutil cp -r mlruns/0/65ac7f76fa744997a460fe8f5facbbba/* gs://pod-seldon-model/pod-toy-model/model
```
or 
```bash
!aws s3 cp mlruns/0/65ac7f76fa744997a460fe8f5facbbba/* s3://bucket
```

# V. Build custom Seldon inference server

Seldon Core offers support for several pre-packaged inference servers, but our model needs to do some preprocessing steps at prediction time (get features from vocab with store_id + product_id)

👉 In fact, there are 3 three ways to do preprocessing

|                    |                                                            | Pros                                                            | Cons                                                                                               |
|--------------------|------------------------------------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Within the model   | Preprocessing code is carried along [with the MLFlow model]([https://](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom))  | Simple, no extra infrastructure is required.                    | Preprocessing steps will be wastefully repeated on each prediction request. Loaded data isnt cached|
| Transform function | Preprocessing code carried along with the inference server | Simple, no extra infrastructure is required. Vocab data is cached | Adds complexity than within the model. Have to build our own inference servers                     |
| Feature Store      | Overkill in our case                                       |                                                                 |                                                                                                    |

➡️ Build custom Seldon inference server solution is chosen


Make custom server image
```bash
!make
```

Test docker image local

```bash
!docker run -it --rm -p 8080:8080 -e PREDICTIVE_UNIT_PARAMETERS='[{"type":"STRING","name":"model_uri","value":"file:///model"}]' -v /home/haunv_it/pod-toy-model/model:/model haunv/mlflowservercustom:latest
```

Push to docker hub (Edit corresponding to your docker hub repositories)

```bash
!docker push haunv/mlflowservercustom:latest
```

# VI. Deploy inference docker image 

1. Add our MLFLOW_SERVER_CUSTOM config to Seldon servers config values

```YAML
#deploy\values.yaml
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
        image: haunv/mlflowservercustom:latest
    
```

2. Register custom ML server config with seldon core

```bash
!git clone seldon-core

!helm upgrade seldon-core seldon-core/helm-charts/seldon-core-operator --namespace seldon-system --values deploy/values.yaml --set istio.enabled=true
    
```

3. Deploy model to Seldon Core on K8S 

YAML file

```YAML
#deploy\seldon-deploy-toy-model.yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: toy-model
  namespace: seldon
spec:
  name: toy-model
  predictors:
  - componentSpecs:
    - spec:
        # We are setting high failureThreshold as installing conda dependencies
        # can take long time and we want to avoid k8s killing the container prematurely
        containers:
        - name: regressor
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

    graph:
      children: []
      implementation: MLFLOW_SERVER_CUSTOM
      modelUri: gs://pod-seldon-model/pod-toy-model/model
      name: regressor
    name: default
    replicas: 1
```

Create K8S namespace

```bash
!kubectl create namespace seldon
```

```bash
!kubectl apply -f deploy/seldon-deploy-toy-model.yaml
```

check Pod status
```bash
!kubectl get pods -n seldon
```
Status when our's pod is ready

```console
NAME                                             READY   STATUS    RESTARTS   AGE
toy-model-default-0-regressor-7d4859989d-265v5   2/2     Running   0          20h
```

- Serving endpoint

`http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/predictions`

  example: http://34.126.90.125/seldon/seldon/toy-model/api/v1.0/predictions

- Example Request predict for store id 2519 & product id 5273
```bash
curl -X POST "http://34.126.157.32/seldon/seldon/toy-model/api/v1.0/predictions" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"data\":{\"names\":[\"store_id\",\"product_id\"],\"ndarray\":[[2519,5273]]}}"
```
- Example Response

model return prediction result is 14.56 for the next 30days

```json
{
  "data": {
    "names": [],
    "ndarray": [
      14.563408949535173
    ]
  },
  "meta": {
    "requestPath": {
      "regressor": "haunv/mlflowservercustom:latest"
    }
  }
}

```

- API document

`http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc`

Example: http://34.126.90.125/seldon/seldon/toy-model/api/v1.0/doc/#/

![apidocument](images/seldon-api-document.png)


# VII. Monitoring dashboard

1. Install Seldon Analytics
  
```bash
!kubectl config set-context $(kubectl config current-context) --namespace=seldon
```

```bash
!helm install seldon-core-analytics seldon-core/helm-charts/seldon-core-analytics  \
        --set grafana_prom_admin_password=password \
        --set persistence.enabled=false \
        --namespace seldon-system \
        --wait

```
note: if get issue *unable to build kubernetes objects from release manifest*..., please downgrade K8S to version 1.21

Map grafana port to 3000

```bash
kubectl port-forward $(kubectl get pods -n seldon-system  -l app.kubernetes.io/name=grafana -o jsonpath='{.items[0].metadata.name}') 3000:3000 -n seldon-system
```
http://localhost:3000

or try with http://34.124.157.160:3000/d/U1cSDzyZz/prediction-analytics?orgId=1&refresh=5s

![grafana](images/grafana.png)

Note: If seldon core is deployed on GKE Google cloud, we need to apply deploy/grafana-map-port.yaml LoadBalancer

# Delete, free resource

```bash
!kubectl delete -f deploy/seldon-deploy-toy-model.yaml
```
```bash
!kubectl delete -f deploy/grafana-map-port.yaml
```

# Additional

**With the architecture designed as above, there are some thing that we can do to improve productivity more in the real environment**

- File distributed system/ database instead of local files
- More monitoring metrics
- Benchmark serving service (must be done to work stably on the production)
- Auto scaling Seldon deployments
- Experiment service : A/B testing
- CI/CD components : Jenkins & Jenkins X
- Data drift detect: Alibi
- Explain (model interpretation): Alibi
