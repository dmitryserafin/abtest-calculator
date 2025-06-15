import requests
import json

data = {
    "a_success": 100,
    "a_total": 1000,
    "b_success": 120,
    "b_total": 1000
}
try:
    response = requests.post("http://localhost:8000/calculate", json=data, timeout=90) # Increased timeout
    print("Request sent. Server response status:", response.status_code)
    # print("Response content:", response.text)
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
