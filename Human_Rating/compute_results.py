import json
from collections import defaultdict, Counter
from fleiss import fleissKappa
import krippendorff
import numpy as np

jiashuo_results = json.load(open("./samples/evaluation_jiashuo.json", "r"))
chunpu_results = json.load(open("./samples/evaluation_chunpu.json", "r"))
zetao_results = json.load(open("./samples/evaluation_zetao.json", "r"))
yuqin_results = json.load(open("./samples/evaluation_yuqin.json", "r"))
order_data = json.load(open("./samples/all_samples.json", "r"))

model_results = {
    "MultiESC": defaultdict(list),
    "KEMI": defaultdict(list),
    "strat": defaultdict(list),
}
inter_results = []
label_map = {"A": 0, "B": 1, "tie": 2}
for idx, line in enumerate(order_data):
    model = line["model"]
    A, B = line["order"]
    model_map = {"A": A, "B": B, "tie": "tie"}

    jiashuo = jiashuo_results[idx]
    chunpu = chunpu_results[idx]
    zetao = zetao_results[idx]
    yuqin = yuqin_results[idx]

    for aspect in {"fluency", "identification", "comforting", "suggestion", "overall"}:
        count = defaultdict(int)
        inter_list = [np.nan] * 4
        inter_list[0] = label_map[jiashuo[aspect]]
        inter_list[1] = label_map[chunpu[aspect]]
        inter_list[2] = label_map[zetao[aspect]]
        try:
            inter_list[3] = label_map[yuqin[aspect]]
        except KeyError:
            pass
        # inter_list = [0] * 3
        # inter_list[label_map[jiashuo[aspect]]] += 1
        # inter_list[label_map[chunpu[aspect]]] += 1
        # inter_list[label_map[zetao[aspect]]] += 1
        inter_results.append(inter_list)
        count[model_map[jiashuo[aspect]]] += 1
        count[model_map[chunpu[aspect]]] += 1
        count[model_map[zetao[aspect]]] += 1
        try:
            count[model_map[yuqin[aspect]]] += 1
        except KeyError:
            pass
        highest = max(count.values())
        final = [k for k, v in count.items() if v == highest]
        if len(final) == 1:
            model_results[model][aspect].append(final[0])
        elif len(final) == 2 and "tie" in final:
            final.remove("tie")
            model_results[model][aspect].append(final[0])
        else:
            model_results[model][aspect].append("tie")

for model in model_results:
    for aspect in model_results[model]:
        print(model, aspect, dict(Counter(model_results[model][aspect])))
    print("************")

print(np.array(inter_results[:75]))
value = krippendorff.alpha(reliability_data=np.transpose(np.array(inter_results)), level_of_measurement='nominal')
print(value)
# kappa = fleissKappa(inter_results[:75], 4)
# print(kappa)
