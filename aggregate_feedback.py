from os.path import join
import numpy as np
import fire
import json

NO_FEEDBACK = {"No", "No Empathy", "MI Non-Adherent"}


def obtain_feedback(sample_dir: str = ""):
    try:
        data = json.load(open(join(sample_dir, "candidates.json"), "r"))
    except FileNotFoundError:
        data = json.load(open(join(sample_dir, "candidates.txt"), "r"))
    try:
        candidate_num = len(data[0]["generation"])
    except KeyError:
        candidate_num = len(data[0]["responses"])
    data_paths = ["feedback_coherence.txt", "feedback_strategy.txt", "feedback_empathy.txt"]
    final_feedback = []
    for path in data_paths:
        path = join(sample_dir, path)
        with open(path, "r") as f:
            lines = f.readlines()
            assert len(lines) == candidate_num * len(data)
            single_feedback = [0 if line.strip() in NO_FEEDBACK else 1 for line in lines]
            if len(final_feedback) == 0:
                final_feedback = single_feedback.copy()
            else:
                final_feedback = [f1 & f2 for f1, f2 in zip(final_feedback, single_feedback)]
    with open(join(sample_dir, "candidate_feedback.npy"), "wb") as f:
        final_feedback = np.array(final_feedback)
        final_feedback.resize((len(data), candidate_num))
        np.save(f, final_feedback)


if __name__ == "__main__":
    fire.Fire(obtain_feedback)
