import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'cement':456.10, 'furnace_slag':56.8, 'fly_ash':87.9, 'water':170.0, 'super_plasticizer':13.5, 'coarse_agg':947.0, 'fine_agg':657.0, 'age':35})

print(r.json())
