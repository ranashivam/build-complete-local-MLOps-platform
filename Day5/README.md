# Kubeflow 1.10 Installation Guide

This guide walks through installing the **Kubeflow 1.10.0** platform using the official Kubeflow manifests, Kustomize, and Kubernetes.

## Prerequisites

Before starting, make sure you have:

- Kubernetes running and accessible through `kubectl`
- Git Bash or a similar Bash-compatible terminal
- PowerShell with Administrator privileges
- Docker installed and running
- Chocolatey installed on Windows
- `kubectl` configured to access your Kubernetes cluster

---

## Step 1: Install Kubeflow 1.10 Manifests

Clone the Kubeflow 1.10.0 manifests repository:

```bash
git clone --branch v1.10.0 https://github.com/kubeflow/manifests.git kubeflow-manifests
```

Move into the cloned directory:

```bash
cd kubeflow-manifests
```

Verify the current Git branch:

```bash
git branch --show-current
```

You should see:

```text
v1.10.0
```

---

## Step 2: Install Kustomize

Open **PowerShell as Administrator** and run:

```powershell
choco install kustomize -y
```

Verify the installation:

```powershell
kustomize version
```

---

## Step 3: Install the Full Kubeflow Platform

Make sure **Step 1** has been completed and that you are inside the `kubeflow-manifests` directory:

```bash
cd kubeflow-manifests
```

Run the following command to build and apply the Kubeflow resources:

```bash
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do
  echo "Retrying to apply resources"
  sleep 20
done
```

The command retries the deployment until the resources are successfully applied.

### Watch the Kubeflow Installation

Open another Git Bash terminal and monitor the Kubernetes pods:

```bash
kubectl get pods -A -w
```

Press `Ctrl+C` to stop watching the pods.

### Troubleshooting: CrashLoopBackOff with KServe and MinIO

If you encounter `CrashLoopBackOff` issues related to KServe or MinIO, pull the required images:

#### Pull the KServe image

```bash
docker pull quay.io/brancz/kube-rbac-proxy:v0.18.0
```

#### Pull the MinIO image

```bash
docker pull kubef/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance
```

#### Update the MinIO Deployment

Run:

```bash
kubectl set image deployment/minio \
  minio=kubef/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance \
  -n kubeflow
```

Then check the pod status again:

```bash
kubectl get pods -n kubeflow
```

---

## Step 4: Check the Important Namespaces

List all Kubernetes namespaces:

```bash
kubectl get namespaces
```

You should see namespaces associated with the Kubeflow installation, including:

```text
cert-manager
istio-system
kubeflow
```

The exact list may vary depending on the cluster and Kubeflow components installed.

---

## Step 5: Check Kubeflow Pods

Check the pods in the main Kubeflow namespaces.

### Kubeflow

```bash
kubectl get pods -n kubeflow
```

### Istio

```bash
kubectl get pods -n istio-system
```

### Cert Manager

```bash
kubectl get pods -n cert-manager
```

Make sure the required pods eventually reach a healthy state, such as `Running` or `Completed`.

---

## Step 6: Open the Full Kubeflow UI

For the full Kubeflow platform, create a local port-forward to the Istio ingress gateway:

```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

Keep this terminal running while you access the Kubeflow UI.

Open your browser and go to:

**http://localhost:8080**

---

## Quick Verification

After installation, these commands are useful for a quick health check:

```bash
kubectl get namespaces
kubectl get pods -n kubeflow
kubectl get pods -n istio-system
kubectl get pods -n cert-manager
```

If the required pods are healthy, start the port-forward:

```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

Then open:

```text
http://localhost:8080
```

## Notes

- Keep the `kubeflow-manifests` directory available if you need to troubleshoot or re-apply the installation.
- The Kubeflow installation command may take some time and can retry several times while Kubernetes resources become ready.
- If pods remain in `Pending`, `CrashLoopBackOff`, or `ImagePullBackOff`, inspect the pod details and logs with `kubectl describe pod` and `kubectl logs`.
