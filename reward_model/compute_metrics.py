from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

predictions, ground_truth = [], []
empathy_pred, empathy_gold = [], []
strategy_pred, strategy_gold = [], []
coherence_pred, coherence_gold = [], []
LABEL_MAP = {
    "Yes": 1,
    "No": 0,
    "Not Sure": 1,
    "No Empathy": 0,
    "Weak Empathy": 1,
    "Strong Empathy": 1,
    "MI Adherent": 1,
    "MI Non-Adherent": 0,
    "Other": 1,
}

# with open("./Llama/test_Llama_tuned.txt", "r") as f:
# with open("./Llama/test_Llama_vanilla.txt", "r") as f:
with open("./Llama/test_Llama.txt", "r") as f:
    # with open("./GPT-4/test_GPT3.5.txt", "r") as f:
    # with open("./GPT-4/test_GPT4.txt", "r") as f:
    for line in f.readlines():
        try:
            pred, gold = line.split(" ## ")
        except:
            pass
        pred = pred.strip()
        gold = gold.strip()
        predictions.append(LABEL_MAP.get(pred, 1))
        ground_truth.append(LABEL_MAP.get(gold, 1))
        if pred in {"Yes", "No"}:
            coherence_gold.append(gold)
            coherence_pred.append(pred)
        elif pred in {"No Empathy", "Weak Empathy", "Strong Empathy"}:
            empathy_gold.append(LABEL_MAP.get(gold, 1))
            empathy_pred.append(LABEL_MAP.get(pred, 1))
        else:
            strategy_gold.append(LABEL_MAP.get(gold, 1))
            strategy_pred.append(LABEL_MAP.get(pred, 1))

print("Overall:")
print("F1: ", f1_score(ground_truth, predictions, average="macro"))
print("Acc: ", accuracy_score(ground_truth, predictions))
print("Recall: ", recall_score(ground_truth, predictions, average="macro"))
print("Precision: ", precision_score(ground_truth, predictions, average="macro"))

print("Empathy:")
print("F1: ", f1_score(empathy_gold, empathy_pred, average="macro"))
print("Acc: ", accuracy_score(empathy_gold, empathy_pred))
print("Recall: ", recall_score(empathy_gold, empathy_pred, average="macro"))
print("Precision: ", precision_score(empathy_gold, empathy_pred, average="macro"))

print("Strategy:")
print("F1: ", f1_score(strategy_gold, strategy_pred, average="macro"))
print("Acc: ", accuracy_score(strategy_gold, strategy_pred))
print("Recall: ", recall_score(strategy_gold, strategy_pred, average="macro"))
print("Precision: ", precision_score(strategy_gold, strategy_pred, average="macro"))

print("Coherence:")
print("F1: ", f1_score(coherence_gold, coherence_pred, average="macro"))
print("Acc: ", accuracy_score(coherence_gold, coherence_pred))
print("Recall: ", recall_score(coherence_gold, coherence_pred, average="macro"))
print("Precision: ", precision_score(coherence_gold, coherence_pred, average="macro"))
