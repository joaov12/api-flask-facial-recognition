import os
import requests

BASE_URL = "http://127.0.0.1:5000"

def compare(img_a, img_b):
    with open(img_a, "rb") as f1, open(img_b, "rb") as f2:
        files = {"image1": f1, "image2": f2}
        response = requests.post(f"{BASE_URL}/compare", files=files)
    return response

def run_test():
    path = os.path.dirname(__file__)
    img1 = os.path.join(path, "exemplo1.jpg")
    img2 = os.path.join(path, "exemplo2.jpg")
    img3 = os.path.join(path, "exemplo3.jpg")

    # Teste 1: exemplo1 vs exemplo2
    resp1 = compare(img1, img2)
    print("== Comparando exemplo1 x exemplo2 ==")
    print("Status:", resp1.status_code)
    print("Resposta:", resp1.json())
    print()

    # Teste 2: exemplo1 vs exemplo3
    resp2 = compare(img1, img3)
    print("== Comparando exemplo1 x exemplo3 ==")
    print("Status:", resp2.status_code)
    print("Resposta:", resp2.json())

if __name__ == "__main__":
    run_test()
