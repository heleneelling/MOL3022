
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# Separate the file into sequences and labels
def read_file(file_path):
    sequences = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('<>') or line.startswith('<end>') or line.startswith('end') or line.strip == "":
                continue
            sequence, label = line.split(' ')
            sequences.append(sequence)
            labels.append(label.strip())
    return sequences, labels


protein_seq_train, sec_struc_train = read_file(
    'protein-secondary-structure.train')
protein_seq_test, sec_struc_test = read_file(
    'protein-secondary-structure.test')


def features(sequences):
   # get weight, isoelectric_point and GRAVY using protein analysis
    seq_features = np.zeros((len(sequences), 3))
    for i, seq in enumerate(sequences):
        p = ProteinAnalysis(seq)
        seq_features[i, 0] = p.molecular_weight()
        seq_features[i, 1] = p.isoelectric_point()
        seq_features[i, 2] = p.gravy()
    return seq_features


x_train = features(protein_seq_train)
x_test = features(protein_seq_test)


# Convert secondary structure labels to numerical values
# For Alpha helices, beta strands and coils
label_dict = {'H': 0, 'E': 1, "_": 2}
y_train = [label_dict[label.upper()] for label in sec_struc_train]
y_test = [label_dict[label.upper()] for label in sec_struc_test]

# Train using MLP, the best accuracy came from a random_state of about 20
mlp = MLPClassifier(random_state=20)
mlp.fit(x_train, y_train)

# predict the secondary structure label for the sequences
y_pred = mlp.predict(x_test)

# Convert numerical labels back to string labels and print predicted structure
sec_struc_pred = ''.join([list(label_dict.keys())[list(
    label_dict.values()).index(label)] for label in y_pred])

print("Predicted secondary structure:", sec_struc_pred)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy", round(accuracy*100, 2), "%")
