import json
import os


def formate_all():
    vanilla_base_path = "../ESConv/DATA/vanilla.vanilla/2023-06-20204748.3e-05.16.1gpu/candidates_10_best_model.bin_test/candidates.json"
    vanilla_muffin_path = "../ESConv/DATA/vanilla.vanilla/2023-06-20204748_muffin_2023-07-26143450/candidates_10_model.bin_test/candidates.json"
    strat_base_path = "../ESConv/DATA/strat.strat/2023-06-20204057.3e-05.16.1gpu/candidates_10_best_model.bin_test/candidates.json"
    strat_muffin_path = "../ESConv/DATA/strat.strat/2023-06-20204057_muffin_2023-07-25191121/candidates_10_model.bin_test/candidates.json"
    KEMI_base_path = "../KEMI/DATA/strat.strat.esconv.sbert/2023-06-30223758.3e-05.16.1gpu/candidates_10_epoch-4.bin_test/candidates.json"
    KEMI_muffin_path = "../KEMI/DATA/strat.strat.esconv.sbert/2023-06-30223758_muffin_2023-07-26204506/candidates_10_model.bin_test/candidates.json"
    TransESC_base_path = "../TransESC/test_base_candidates/candidates.txt"
    TransESC_muffin_path = "../TransESC/test_muffin_candidates/candidates.txt"
    MultiESC_base_path = "../MultiESC/final_output/lwg_whlookahead_generate_candidate_10/test_candidates.json"
    MultiESC_muffin_path = "../MultiESC/mitigate_output/MultiESC_mitigate_2023-07-29170059/checkpoint-100_candidate_10/test_candidates.json"

    output_dir = "./output_candidates"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ########### Vanilla Base ###########
    vanilla_base_data = json.load(open(vanilla_base_path, "r"))
    output_data = []
    context = []
    last_sample_id = -1
    for idx, model in enumerate(vanilla_base_data):
        if last_sample_id != model["sample_id"]:
            context = []
            last_sample_id = model["sample_id"]
        context.append(model["post"])
        output_data.append({
            "sample_id": model["sample_id"],
            "context": context[-3:],
            "response": model["generation"]
        })
        context.append(model["response"])
    output_path = os.path.join(output_dir, "vanilla_base.json")
    json.dump(output_data, open(output_path, "w"), indent=4)
    ########### Vanilla Muffin ###########
    vanilla_muffin_data = json.load(open(vanilla_muffin_path, "r"))
    output_data = []
    context = []
    last_sample_id = -1
    for idx, model in enumerate(vanilla_muffin_data):
        if last_sample_id != model["sample_id"]:
            context = []
            last_sample_id = model["sample_id"]
        context.append(model["post"])
        output_data.append({
            "sample_id": model["sample_id"],
            "context": context[-3:],
            "response": model["generation"]
        })
        context.append(model["response"])
    output_path = os.path.join(output_dir, "vanilla_muffin.json")
    json.dump(output_data, open(output_path, "w"), indent=4)
    ########### Strat Base ###########
    strat_base_data = json.load(open(strat_base_path, "r"))
    output_data = []
    context = []
    last_sample_id = -1
    for idx, model in enumerate(strat_base_data):
        if last_sample_id != model["sample_id"]:
            context = []
            last_sample_id = model["sample_id"]
        context.append(model["post"])
        output_data.append({
            "sample_id": model["sample_id"],
            "context": context[-3:],
            "response": model["generation"]
        })
        context.append(model["response"])
    output_path = os.path.join(output_dir, "strat_base.json")
    json.dump(output_data, open(output_path, "w"), indent=4)
    ########### Strat Muffin ###########
    strat_muffin_data = json.load(open(strat_muffin_path, "r"))
    output_data = []
    context = []
    last_sample_id = -1
    for idx, model in enumerate(strat_muffin_data):
        if last_sample_id != model["sample_id"]:
            context = []
            last_sample_id = model["sample_id"]
        context.append(model["post"])
        output_data.append({
            "sample_id": model["sample_id"],
            "context": context[-3:],
            "response": model["generation"]
        })
        context.append(model["response"])
    output_path = os.path.join(output_dir, "strat_muffin.json")
    json.dump(output_data, open(output_path, "w"), indent=4)
    ########### KEMI Base ###########
    KEMI_base_data = json.load(open(KEMI_base_path, "r"))
    output_data = []
    context = []
    last_sample_id = -1
    for idx, model in enumerate(KEMI_base_data):
        if last_sample_id != model["sample_id"]:
            context = []
            last_sample_id = model["sample_id"]
        context.append(model["post"])
        output_data.append({
            "sample_id": model["sample_id"],
            "context": context[-3:],
            "response": model["generation"]
        })
        context.append(model["response"])
    output_path = os.path.join(output_dir, "KEMI_base.json")
    json.dump(output_data, open(output_path, "w"), indent=4)
    ########### KEMI Muffin ###########
    KEMI_muffin_data = json.load(open(KEMI_muffin_path, "r"))
    output_data = []
    context = []
    last_sample_id = -1
    for idx, model in enumerate(KEMI_muffin_data):
        if last_sample_id != model["sample_id"]:
            context = []
            last_sample_id = model["sample_id"]
        context.append(model["post"])
        output_data.append({
            "sample_id": model["sample_id"],
            "context": context[-3:],
            "response": model["generation"]
        })
        context.append(model["response"])
    output_path = os.path.join(output_dir, "KEMI_muffin.json")
    json.dump(output_data, open(output_path, "w"), indent=4)
    ########### TransESC Base ###########
    TransESC_base_data = json.load(open(TransESC_base_path, "r"))
    output_data = []
    for idx, model in enumerate(TransESC_base_data):
        output_data.append({
            "sample_id": model["sample_idx"],
            "context": model["context"],
            "response": model["generation"]
        })
    output_path = os.path.join(output_dir, "TransESC_base.json")
    json.dump(output_data, open(output_path, "w"), indent=4)
    ########### TransESC Muffin ###########
    TransESC_muffin_data = json.load(open(TransESC_muffin_path, "r"))
    output_data = []
    for idx, model in enumerate(TransESC_muffin_data):
        output_data.append({
            "sample_id": model["sample_idx"],
            "context": model["context"],
            "response": model["generation"]
        })
    output_path = os.path.join(output_dir, "TransESC_muffin.json")
    json.dump(output_data, open(output_path, "w"), indent=4)
    ########### MultiESC Base ###########
    MultiESC_base_data = json.load(open(MultiESC_base_path, "r"))
    output_data = []
    for idx, model in enumerate(MultiESC_base_data):
        output_data.append({
            "sample_id": model["sample_id"],
            "context": model["context"],
            "response": model["responses"]
        })
    output_path = os.path.join(output_dir, "MultiESC_base.json")
    json.dump(output_data, open(output_path, "w"), indent=4)
    ########### MultiESC Muffin ###########
    MultiESC_muffin_data = json.load(open(MultiESC_muffin_path, "r"))
    output_data = []
    for idx, model in enumerate(MultiESC_muffin_data):
        output_data.append({
            "sample_id": model["sample_id"],
            "context": model["context"],
            "response": model["responses"]
        })
    output_path = os.path.join(output_dir, "MultiESC_muffin.json")
    json.dump(output_data, open(output_path, "w"), indent=4)


if __name__ == "__main__":
    formate_all()
