import requests

url = 'http://127.0.0.1:5000/predict_api'

r = requests.post(url, json={'Age (in days) :': 28,
                             'Cement (in kg) : ': 540.0,
                             'Water (in kg) :': 162.0,
                             'Fly ash (in kg) :': 0.0,
                             'Blast furnace slag (in kg) ': 0.0,
                             'Super plasticizer (in kg) :': 2.5})

print(r.json())
