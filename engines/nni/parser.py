import json
from collections import defaultdict

json_file_path = "./experiment_random.json"

choices = ["layerchoice_cell__0_1_", "layerchoice_cell__0_2_", "layerchoice_cell__1_2_", "layerchoice_cell__0_3_", "layerchoice_cell__1_3_", "layerchoice_cell__2_3_"]

trial_count = defaultdict(int)
trial_idx = []

with open(json_file_path, 'r') as j:
    contents = json.loads(j.read())
    trials = contents.get("trialMessage")
    count = 0
    for i, trial in enumerate(trials):
        if trial["status"] == "SUCCEEDED":
            hp = json.loads(trial["hyperParameters"][0])
            if trial_count[hp["parameter_id"]] != 0:
                continue
            else:
                trial_count[hp["parameter_id"]] += 1
                script = hp["parameters"]["model_script"]
                config = []
                for choice in choices:
                    result = script.find(choice)
                    config.append(script[result: result+len(choice)+18].split()[0].split("_", 5)[-1])
                config = '-'.join(config)
                print(hp["parameter_id"], config)
                # print(trial["finalMetricData"])
            count += 1
        # if count > 200:
        #     break
    print(len(trial_count))
    print(count)
