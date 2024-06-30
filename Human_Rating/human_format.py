import json

data = json.load(open("./samples/all_samples.json", "r"))
context_dict = {
    "MultiESC": json.load(open("../MultiESC/mitigate_output/test_dataset.json", "r")),
    "strat": json.load(open("../ESConv/context.json")),
    "KEMI": json.load(open("../ESConv/context.json")),
}


def json_version():
    processed_data = []
    for idx, line in enumerate(data):
        model = line["model"]
        index = line["index"]
        temp_dict = {"idx": idx}
        try:
            temp_dict["situation"] = "The seeker has the following problem: " + context_dict[model][index]["situation"]
        except KeyError:
            temp_dict["situation"] = "The seeker has the following problem: " + context_dict[model][index]["situations"]
        temp_dict["context"] = context_dict[model][index]["context"][-5:]
        temp_dict["Supporter AB"] = line["AB"]
        temp_dict["options"] = "A, B, tie"
        temp_dict["fluency"] = ""
        temp_dict["identification"] = ""
        temp_dict["comforting"] = ""
        temp_dict["suggestion"] = ""
        temp_dict["overall"] = ""
        processed_data.append(temp_dict)

    json.dump(processed_data, open("./samples/samples4evaluation.json", "w"), indent=2)


json_version()
