import json
with open("local_evaluations/results/mirix_HomeEnv/0/results.json", "r") as f:
    results = json.load(f)

for result in results:
    print("Question: ", result["question"])
    print("Answer: ", result["answer"])
    print("Response: ", result["response"])
    print("--------------------------------")