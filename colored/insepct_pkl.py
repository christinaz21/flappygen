import pickle

with open("data/flappy_bird/collected/0a1e6cf3-1faf-4e2b-ac87-0f4590b7f03f.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print("Data: ", data)

if isinstance(data, dict):
    print(data.keys())
    print("First few actions:", data.get("actions", [])[:5])
else:
    print("First few elements:", data[:5])
