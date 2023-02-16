import json

data = json.load(open('res_1epoch.json', 'r'))


with open('eval_fixed.jsonl', 'w') as w:
    for i, a in data.items():
        w.write(json.dumps({'id': i, 'prediction': a[0]['generated_text']}) + '\n')
