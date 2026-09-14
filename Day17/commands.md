# ============================================================
# 1. Build the Docker image
# ============================================================
# Build the Docker image using the Dockerfile in the current folder.
# The image is tagged with the Docker Hub repository and version v17.

docker build -t shivamrana28/iris-m1-app:v17 .


# ============================================================
# 2. Push the image to Docker Hub
# ============================================================
# Upload the Docker image to Docker Hub.
# Make sure you are logged in first using: docker login

docker push shivamrana28/iris-m1-app:v17


# ============================================================
# 3. Load the image into the Kind Kubernetes cluster
# ============================================================
# Import the Docker image into the local Kind cluster.
# This allows Kubernetes to use the image without pulling it
# from Docker Hub.

kind load docker-image shivamrana28/iris-m1-app:v17


# ============================================================
# 4. Open the Kubeflow dashboard
# ============================================================
# Forward the Istio gateway service to localhost port 8080.
# Keep this terminal running while accessing Kubeflow.

kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80

# Kubeflow URL:
# http://localhost:8080

# Kubeflow login credentials:
# Username: user@example.com
# Password: 12341234


# ============================================================
# 5. Open the MLflow dashboard
# ============================================================
# Forward the MLflow service to localhost port 5000.
# Keep this terminal running while accessing MLflow.

kubectl port-forward svc/mlflow-service -n kubeflow 5000:5000

# MLflow URL:
# http://localhost:5000


# ============================================================
# 6. Monitor Kubeflow pipeline pods
# ============================================================
# Watch the pipeline pods in the Kubeflow user namespace.
# The command continuously refreshes as pod statuses change.
# Press Ctrl+C to stop watching.

kubectl get pods -n kubeflow-user-example-com -w