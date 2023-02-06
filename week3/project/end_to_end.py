import requests
import json

def run_end_to_end_tests():
    with open('data/requests.json') as req:
        for example in req.readlines():
            r = requests.post('http://localhost:8000/predict', json=json.loads(example))
            print(r.json())

if __name__ == '__main__':
    run_end_to_end_tests()