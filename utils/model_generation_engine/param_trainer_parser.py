'''
Parses the Parameter_Tuning_Reports.txt file so that it can automatically set the parameters in the continue_training method of the data_processor file.
'''
from .. import common
from typing import List


def extract_datum(reports: str, key: str, terminator: str) -> str:
    start_ind = reports.find(key) + len(key)
    end_ind = reports.find(terminator, start_ind)

    return reports[start_ind : end_ind].strip()



def parse_reports(coin: str, model_architecture: str) -> List[dict]:
    try:
        with open(f"reports/{coin}_Parameter_Tuning_Report_{model_architecture}.txt", 'r') as f:
            reports = f.read()
    except:
        print(f"reports/{coin}_Parameter_Tuning_Report_{model_architecture}.txt not found.")
        raise

    models = []
    while len(reports) > 10:
        model = {}
        model["model_num"] =  int(extract_datum(reports, "MODEL:", '\n'))
        model["architecture"] = extract_datum(reports, "PARAMETERS:", 'e')
        model["eta"] = float(extract_datum(reports, "eta:", '|'))
        model["decay"] = float(extract_datum(reports, "decay:", '|'))
        model["dropout"] = float(extract_datum(reports, "dropout:", '\n'))
        model["accuracy"] = float(extract_datum(reports, "Decision:", '\n'))
        model["inaccuracy"] = float(extract_datum(reports, "Opposite:", '\n'))

        # Add model if meets accuracy threshhold
        if model["accuracy"] > common.PROMISING_ACCURACY_THRESHOLD and model["inaccuracy"] < common.INACCURACY_THRESHOLD:
            models.append(model)

        # delete up until next model
        start_ind = reports.find("MODEL", reports.find("Opposite"))
        reports = reports[start_ind:]

    return models



def list_promising_model_details(coin: str, model_architecture: str) -> None:
    models = parse_reports(coin, model_architecture)

    count = 0
    for model in models:
        count += 1
        print(f"Model num: {model['model_num']}")
        print(f"Model acc: {model['accuracy']}")
        print(f"Model bad: {model['inaccuracy']}")
        print()

        print(f"{count} promising models found.")



def get_model_params(coin: str, filename: str) -> dict:
    start_ind = filename.find('_') + 1
    end_ind = filename.find('_', start_ind) + 2
    model_architecture = filename[start_ind:end_ind]

    start_ind = filename.find('_', end_ind) + 1
    end_ind = filename.find('_', start_ind)
    model_num = filename[start_ind:end_ind]

    try:
        models = parse_reports(coin, model_architecture)
        for model in models:
            if model["model_num"] == int(model_num):
                return model
    except:
        return None



if __name__ == "__main__":
    list_promising_model_details("all", "Pi_3")
