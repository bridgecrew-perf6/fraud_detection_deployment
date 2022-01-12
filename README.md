# Fraud Detection classification with Kubernetes files for deployment

### Information
In this project I used a classic dataset for Fraud Detection that can be found [here](https://www.kaggle.com/mlg-ulb/creditcardfraud). I used a Logistic regressiion model for classification, RESTApi, Docker and Kubernetes for deployment.  
<b>NOTE:</b> The Dockerfile also copies the [train.py](https://github.com/) file, which is not used. I included it regardless for future work implementation (see Future Work).


### Instructions for deployment (assuming you have "minikube" installed)
I used Windows Power Shell for the below instructions, and Cygwin command line for using "curl" commands.  

From inside the project folder, do the following:
1.  `docker build -t cc-fraud:0.1 .` : Builds Docker image
2.  `minikube start` : Starts the Kubernetes cluster
3.  Create Deployment and Service to expose the Pods outside of the cluster.
    1. `kubectl create --filename deploy.yaml`
    2. `kubectl create --filename service.yaml`
4. Load local Docker image on Minikube cluster
    1. `minikube image load cc-fraud:0.1`
5. Generate a network tunnel from the host machine to the Minikube cluster.
    1. `minikube tunnel`
6. Get external IP address
    1. `kubectl get services`
7. Now we can use `<external-ip>:<port>` as a link to send requests to the app from outside the cluster.

In my case, the external ip is: 127.0.0.1, therefore I can use `127.0.0.1:5000` to query predictions.

Below are some examples to query from the command line (data is randomly chosen from dataset):  


`curl -X POST http://127.0.0.1:5000/predict -d '{"data": [40981.0, -0.7842910186562471, -2.7012257535647324, -0.0967386127210913, 0.521375119473271, -1.70573851640263, 1.10351855890133, 0.431322198687247, 0.22802116977057896, 1.4652977928665, -1.3066944304644599, 1.33128250555487, 1.57790719991889, -0.440560604628892, 0.128074950688203, 0.0286489136111531, -0.9625872691461059, 0.519210020956086, -0.38704390740026795, -0.0490091299964628, 0.6502217079234472, 0.516072072435101, -0.338177074515334, -0.6258812915555879, -0.22207187222957603, -0.0298968921431093, -0.6783181524614119, -0.0659639052697792, 0.18338004089478102, 185.375]}' -H 'Content-Type: application/json'` (Non-Fraud, so it's supposed to predict "0")

`curl -X POST http://127.0.0.1:5000/predict -d '{"data": [26899.0, -4.263979957375013, 2.901188080680856, -3.7646449731967673, 3.124319110867031, -2.6429012689627394, -2.5177651329422983, -2.236984407382579, 1.1275019172451173, -2.504517212462476, -2.019373365934702, 2.991421924993211, -1.9409590671956645, 0.461807829323489, -1.802834475317419, -0.0228148529460228, -1.952416410433839, -1.8082765194756663, -1.997969244031534, 1.10435473489394, 0.6502217079234472, 0.8079413701109316, -1.57905510076487, -0.6258812915555879, 0.13456529188962, 1.3528964721899515, -0.22267085685559398, 0.3339816451473099, 0.2749172881886554, 99.99]}' -H 'Content-Type: application/json'` (Fraud, so it's supposed to predict "1")


### Versions:
python = '3.7.4'  
Flask==2.0.1  
imbalanced_learn==0.8.1  
imblearn==0.0  
pandas==1.1.5  
scikit_learn==1.0.2  

### Future Work:
- [ ] Scheldule model re-trains with Kubernetes while maintaining zero down time.

### References for Kubernetes deployment
https://kubernetes.io/docs/tutorials/kubernetes-basics/  
https://thecodinginterface.com/blog/flask-rest-api-minikube/  
https://blog.dataiku.com/how-to-perform-basic-ml-serving-with-python-docker-kubernetes


