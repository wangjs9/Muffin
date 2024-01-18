empathy, strategy, coherence = [], [], []

with open("./consistency.txt", "r") as fp:
    lines = fp.readlines()
    for line in lines:
        lists = line.strip().split(",")
        for alist in lists:
            score1, score2, score3 = alist.split()
            empathy.append(int(score1))
            strategy.append(int(score2))
            coherence.append(int(score3))

KEMI_real_emp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
KEMI_real_str = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
KEMI_real_coh = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

KEMI_base_emp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
KEMI_base_str = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
KEMI_base_coh = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

KEMI_muffin_emp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
KEMI_muffin_str = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
KEMI_muffin_coh = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

AI_empathy, AI_strategy, AI_coherence = [], [], []
for i in range(len(KEMI_base_emp)):
    AI_empathy.append(KEMI_real_emp[i])
    AI_empathy.append(KEMI_base_emp[i])
    AI_empathy.append(KEMI_muffin_emp[i])

for i in range(len(KEMI_base_str)):
    AI_strategy.append(KEMI_real_str[i])
    AI_strategy.append(KEMI_base_str[i])
    AI_strategy.append(KEMI_muffin_str[i])

for i in range(len(KEMI_base_coh)):
    AI_coherence.append(KEMI_real_coh[i])
    AI_coherence.append(KEMI_base_coh[i])
    AI_coherence.append(KEMI_muffin_coh[i])

human_ratings = empathy + strategy + coherence
AI_ratings = AI_empathy + AI_strategy + AI_coherence

x = [int(x == y) for x, y in zip(human_ratings, AI_ratings)]
print(sum(x) / len(x))
