import pandas as pd
import numpy as np
from scipy import stats
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random
from typing import List, Tuple


def shuffle_data(features: List[List[float]], targets: List[int]) -> Tuple[List[List[float]], List[float]]:
    '''
    Used for shuffling the data during the training phase.
    '''
    size = len(features)
    for row_ind in range(size):
        swap_row_ind = random.randrange(size)

        feat_tmp_row = features[swap_row_ind]
        tar_tmp_row = targets[swap_row_ind]

        features[swap_row_ind] = features[row_ind]
        targets[swap_row_ind] = targets[row_ind]

        features[row_ind] = feat_tmp_row
        targets[row_ind] = tar_tmp_row

    return features, targets



def generate_dataset(data: pd.DataFrame, limit: int, offset: int, data_aug_per_sample: int = 0, shuffle: bool = False) -> Tuple[List[List[float]], List[float]]:
    '''
    Returns a list of tuples, of which the first element of the tuple is the list of values for the features and the second is the target value
    NOTES:
        - data_aug_per_sample param determines how many extra datapoints to generate per each original datapoint * its frequency metric (i.e., signal_ratios)
        - signal_ratios variable is used to upsample underrepresented categories more than their counterparts when augmenting the data
    '''
    # to determine relative frequency of signals
    new_data = data.iloc[:limit,:]
    vals = new_data["signal"].value_counts().sort_index()
    signal_ratios = [vals.max()/x for x in vals]

    features = []
    targets = []
    for row in range(offset, limit):
        target = data.iloc[row, -1]
        feature_set = data.iloc[row, 0:7].values.tolist()
        features.append(feature_set)
        targets.append(target)

        # this evens out the datapoints per category
        for i in range(data_aug_per_sample * round(signal_ratios[target])):
            row_info_aug = []
            for feature in range(len(feature_set)):
                rand_factor = 1 + random.uniform(-0.000001, 0.000001)
                row_info_aug.append(data.iloc[row, feature] * rand_factor)
            features.append(row_info_aug)
            targets.append(target)

    if shuffle:
        features, targets = shuffle_data(features, targets)

    return features, targets



def get_datasets(coin: str, data_aug_factor: int = 0) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    '''
    Splits dataset into training and testing datasets.
    NOTE: uses no data augmentation by default and will only apply data_aug_factor to the training dataset.
    '''
    # Load data
    data = pd.read_csv(f"datasets/complete/{coin}_historical_data_complete.csv")
    data = data.drop(columns=["date"])
    data["signal"] = data["signal"].astype("int64")

    print("Checking for anomalous data...")
    # Check for any anomalous, unnormalized data in all columns except the signal column
    for c in range(len(data.columns)-1):
        for r in range(len(data)):
            if (data.iloc[r, c] > 1) == True:
                print(f"ERROR: Unnormalized data still present in the dataset. Column: {data.columns[c]} | Row: {r} | Value: {data.iloc[r, c]}")
                data.iloc[r, c] = data.iloc[r, c] / 1000

    # Split into training, validation, testing
    # 70-15-15 split
    n_datapoints = data.shape[0]
    train_end = int(round(n_datapoints*0.75))

    print("Creating training data...")
    X_train, y_train = generate_dataset(data, train_end, 0, data_aug_factor, shuffle=True)
    print("Creating testing data...")
    X_test, y_test = generate_dataset(data, n_datapoints, train_end)

    return X_train, y_train, X_test, y_test



def save_confusion_matrix_heatmap(y_test, preds):
	matrix = confusion_matrix(y_test, preds)
	matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

	# Build the plot
	plt.figure(figsize=(16,7))
	sns.set(font_scale=1.4)
	sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

	# Add labels to the plot
	class_names = ["BUY X", "HODL", "SELL Y"]
	tick_marks = np.arange(len(class_names))
	tick_marks2 = tick_marks + 0.5
	plt.xticks(tick_marks, class_names, rotation=25)
	plt.yticks(tick_marks2, class_names, rotation=0)
	plt.xlabel('Predicted label')
	plt.ylabel('True label')
	plt.title('Confusion Matrix for Random Forest Model')
	plt.savefig("x.png")




print("Generating dataset...")
X_train, y_train, X_test, y_test = get_datasets("bitcoin", 32)
print("Dataset generated")

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


mode = stats.mode(y_train)[0]
base_preds = [mode for _ in range(len(y_train))]
count = 0
for i, x in enumerate(y_test):
	if x == base_preds[i]:
		count += 1

print(f"Baseline accuracy by choosing mode: {count/len(y_test)}")

rf = RandomForestClassifier(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds)}")

save_confusion_matrix_heatmap(y_test, preds)
print(classification_report(y_test, preds))

joblib.dump(rf, "models/RFClassifier.joblib")
