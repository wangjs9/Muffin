import json
import os
import pickle

context_dict = {
    "MultiESC": json.load(open("output_candidates/MultiESC_base.json", "r")),
    "strat": json.load(open("output_candidates/strat_base.json", "r")),
    "vanilla": json.load(open("output_candidates/vanilla_base.json", "r")),
    "KEMI": json.load(open("output_candidates/KEMI_base.json", "r")),
    "TransESC": json.load(open("output_candidates/TransESC_base.json", "r")),
}


def self_disclosure():
    data_path = "../ESConv/DATA/strat.strat/2023-06-20204057.3e-05.16.1gpu/inference_results/test_generations.json"
    # data_path = "../MultiESC/final_output/lwg_whlookahead_generate/inference_results/MultiESC_test_predictions_beam4.txt"
    # data_path = "../ESConv/DATA/vanilla.vanilla/2023-06-20204748.3e-05.16.1gpu/inference_results/test_generations.json"

    if data_path.endswith(".json"):
        conversations = json.load(open(data_path, "r"))
    else:
        conversations = []
        with open(data_path, "r") as file:
            for line in file.readlines():
                conversations.append(line.strip())
    counter = 0
    total_response = 0
    for conversation in conversations:
        total_response += 1
        try:
            generation = conversation["generation"]
        except:
            generation = conversation
        if "i can understand" in generation:
            counter += 1
        elif "i understand" in generation:
            counter += 1
        elif "i also have" in generation:
            counter += 1
        elif "in the same" in generation:
            counter += 1
        elif "similar" in generation:
            counter += 1
        elif "i have been " in generation:
            counter += 1
        elif "hard situation" in generation:
            counter += 1
    print(counter / total_response)


def TransESC_data():
    other_train_path = "../ESConv/DATA/strat.strat/2023-06-20204057.3e-05.16.1gpu/candidates_10_best_model.bin_train/feedback_coherence.txt"
    train_path = "../TransESC/dataset/train_csk.pkl"
    test_path = "../TransESC/dataset/test_csk.pkl"

    train_data = pickle.load(open(train_path, "rb"))
    test_data = pickle.load(open(test_path, "rb"))
    with open(other_train_path, "r") as file:
        other_train_data = file.readlines()
    print(len(train_data))
    print(len(test_data))
    print(len(other_train_data) / 10)


def MultiESC_generation():
    base_path = "../MultiESC/final_output/lwg_whlookahead_generate/inference_results/MultiESC_test_predictions_beam4.txt"
    muffin_path = "../MultiESC/mitigate_output/MultiESC_mitigate_2023-07-29170059/checkpoint-100/inference_results/mitigate_test_predictions_beam4.txt"
    context = context_dict["MultiESC"]
    with open(base_path, "r") as file:
        base_responses = file.readlines()
        base_responses = [line.strip() for line in base_responses]
    with open(muffin_path, "r") as file:
        muffin_responses = file.readlines()
        muffin_responses = [line.strip() for line in muffin_responses]
    base_generation = []
    for base, muffin, line in zip(base_responses, muffin_responses, context):
        base_generation.append({
            "sample_id": line["sample_id"],
            "context": line["context"],
            "MultiESC base": base,
            "MultiESC muffin": muffin,
        })
    output_path = "../MultiESC/final_output/generations.json"
    with open(output_path, "w") as file:
        json.dump(base_generation, file, indent=4)
    txt_output_path = "../MultiESC/final_output/generations.txt"
    with open(txt_output_path, "w") as file:
        for line in base_generation:
            file.write(json.dumps(line) + "\n")


def format_all():
    vanilla_base_path = "../ESConv/DATA/vanilla.vanilla/2023-06-20204748.3e-05.16.1gpu/inference_results/test_generations.json"
    vanilla_muffin_path = "../ESConv/DATA/vanilla.vanilla/2023-06-20204748_muffin_2023-07-26143450/inference_results/test_generations.json"
    strat_base_path = "../ESConv/DATA/strat.strat/2023-06-20204057.3e-05.16.1gpu/inference_results/test_generations.json"
    strat_muffin_path = "../ESConv/DATA/strat.strat/2023-06-20204057_muffin_2023-07-25191121/inference_results/test_generations.json"
    KEMI_base_path = "../KEMI/DATA/strat.strat.esconv.sbert/2023-06-30223758.3e-05.16.1gpu/res_epoch-4.bin_test_k.30_p.0.3_b.1_t.0.7_lp.1.0_rp.1.0_ng.3/gen.json"
    KEMI_muffin_path = "../KEMI/DATA/strat.strat.esconv.sbert/2023-06-30223758_muffin_2023-07-26204506/res_model.bin_test_k.30_p.0.3_b.1_t.0.7_lp.1.0_rp.1.0_ng.3/gen.json"
    TransESC_real_path = "../TransESC/generated_data/ref_strategy.json"
    TransESC_base_path = "../TransESC/generated_data/summary.txt"
    TransESC_muffin_path = "../TransESC/mitigation_data/summary.txt"
    output_dir = "./output"
    context = context_dict["vanilla"]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ######### process vanilla and strat #########
    vanilla_base_data = json.load(open(vanilla_base_path, "r"))
    vanilla_muffin_data = json.load(open(vanilla_muffin_path, "r"))
    strat_base_data = json.load(open(strat_base_path, "r"))
    strat_muffin_data = json.load(open(strat_muffin_path, "r"))
    output_data = []
    assert len(vanilla_base_data) == len(vanilla_muffin_data) == len(strat_base_data) == len(strat_muffin_data) == len(
        context)
    for idx, (vanilla_base, vanilla_muffin, strat_base, strata_muffin) \
            in enumerate(zip(vanilla_base_data, vanilla_muffin_data, strat_base_data, strat_muffin_data)):
        # response = vanilla_base["response"]
        vanilla_base_generation = vanilla_base["generation"]
        vanilla_muffin_generation = vanilla_muffin["generation"]
        strat_base_generation = strat_base["generation"]
        strat_muffin_generation = strata_muffin["generation"]
        output_data.append({
            "sample_id": vanilla_base["sample_id"],
            "context": context[idx]["context"],
            # "response": response,
            "vanilla base": vanilla_base_generation,
            "vanilla muffin": vanilla_muffin_generation,
            "strat base": strat_base_generation,
            "strat muffin": strat_muffin_generation,
        })
    output_path = os.path.join(output_dir, "vanilla_strat.json")
    json.dump(output_data, open(output_path, "w"), indent=4)

    ######### process KEMI #########
    KEMI_base_data = json.load(open(KEMI_base_path, "r"))
    KEMI_muffin_data = json.load(open(KEMI_muffin_path, "r"))
    output_data = []
    context = context_dict["KEMI"]
    assert len(KEMI_base_data) == len(KEMI_muffin_data) == len(context)
    for idx, (KEMI_base, KEMI_muffin) in enumerate(zip(KEMI_base_data, KEMI_muffin_data)):
        # response = KEMI_base["response"]
        KEMI_base_generation = KEMI_base["generation"]
        KEMI_muffin_generation = KEMI_muffin["generation"]
        output_data.append({
            "sample_id": KEMI_base["sample_id"],
            "context": context[idx]["context"],
            # "response": response,
            "KEMI base": KEMI_base_generation,
            "KEMI muffin": KEMI_muffin_generation,
        })
    output_path = os.path.join(output_dir, "KEMI.json")
    json.dump(output_data, open(output_path, "w"), indent=4)

    ######### process TransESC #########
    TransESC_real = json.load(open(TransESC_real_path, "r"))
    with open(TransESC_base_path, "r") as file:
        TransESC_base = file.readlines()
    with open(TransESC_muffin_path, "r") as file:
        TransESC_muffin = file.readlines()
    output_data = []
    context = context_dict["TransESC"]
    assert len(TransESC_real) == len(TransESC_base) == len(TransESC_muffin) == len(context)
    for idx, (real, base, muffin) in enumerate(zip(TransESC_real, TransESC_base, TransESC_muffin)):
        base_response = base.split("\t")[-1]
        muffin_response = muffin.split("\t")[-1]
        output_data.append({
            # "sample_id": real["sample_id"],
            "context": [context[idx]["context"]],
            # "response": real.strip(),
            "TransESC base": base_response,
            "TransESC muffin": muffin_response,
        })
    output_path = os.path.join(output_dir, "TransESC.json")
    json.dump(output_data, open(output_path, "w"), indent=4)


def format_ablation():
    data_path = {
        "coherence": "../ESConv/DATA/strat.strat/2023-06-20204057_muffin_coherence/mitigate_strat_600/inference_results/test_generations.json",
        "empathy": "../ESConv/DATA/strat.strat/2023-06-20204057_muffin_empathy/mitigate_strat_400/inference_results/test_generations.json",
        "strategy": "../ESConv/DATA/strat.strat/2023-06-20204057_muffin_strategy/mitigate_strat_600/inference_results/test_generations.json",
    }
    context = context_dict["vanilla"]
    output_data = []
    for model, path in data_path.items():
        data = json.load(open(path, "r"))
        for idx, (d, c) in enumerate(zip(data, context)):
            output_data.append({
                "sample_id": d["sample_id"],
                "context": c["context"],
                "response": d["response"],
                "generation": d["generation"],
            })
        output_path = f"./output/strat_{model}.json"
        json.dump(output_data, open(output_path, "w"), indent=4)


# MultiESC_generation()
format_all()

# format_ablation()
