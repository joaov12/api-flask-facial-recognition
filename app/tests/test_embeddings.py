import os
import requests

BASE_URL = "http://127.0.0.1:5000"

def run_test():
    image_path = os.path.join(os.path.dirname(__file__), "exemplo1.jpg")
    with open(image_path, "rb") as img:
        files = {"image": img}
        response = requests.post(f"{BASE_URL}/embeddings", files=files)

    print("Status:", response.status_code)
    print("Resposta embeddings:", response.json())

if __name__ == "__main__":
    run_test()
