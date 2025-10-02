import requests

API_URL = "http://localhost:8000"
API_KEY = "dev-key-12345"
headers = {"X-API-Key": API_KEY}

print("=" * 60)
print("API Testing")
print("=" * 60)

# Test 1: Health check
print("\n1. Health Check")
response = requests.get(f"{API_URL}/health")
print(f"   Status: {response.status_code}")
print(f"   Result: {response.json()}")

# Test 2: Upload document
print("\n2. Upload Document")
with open("documents/test.txt", "rb") as f:
    files = {"files": ("test.txt", f, "text/plain")}
    response = requests.post(
        f"{API_URL}/api/v1/documents/upload",
        headers=headers,
        files=files
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Uploaded: {result['files_uploaded']} file(s)")
    else:
        print(f"   Error: {response.text}")

# Test 3: Process uploaded documents
print("\n3. Process Documents")
data = {
    "task_types": ["qa_generation", "summarization"],
    "quality_filter": True
}
response = requests.post(
    f"{API_URL}/api/v1/processing/process-uploaded",
    headers=headers,
    json=data
)
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"   Dataset ID: {result['dataset_id']}")
    print(f"   Examples: {result['examples_generated']}")
    print(f"   Quality: {result['quality_score']:.2f}")
    dataset_id = result['dataset_id']
else:
    print(f"   Error: {response.text}")
    dataset_id = None

# Test 4: List datasets
print("\n4. List Datasets")
response = requests.get(
    f"{API_URL}/api/v1/datasets/list",
    headers=headers
)
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"   Total datasets: {result['total']}")

# Test 5: Export dataset
if dataset_id:
    print("\n5. Export Dataset")
    data = {"format": "jsonl", "split_data": True}
    response = requests.post(
        f"{API_URL}/api/v1/datasets/{dataset_id}/export",
        headers=headers,
        json=data
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Exported to: {result['export_path']}")

print("\n" + "=" * 60)
print("Tests Complete!")
print("=" * 60)