import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tqdm
import joblib
from collections import OrderedDict


def preprocess_data(data_file, output_dir):
    """
    Exercice : Fonction pour prétraiter les données brutes et les préparer pour l'entraînement de modèles.

    Objectifs :
    1. Charger les données brutes à partir d’un fichier CSV.
    2. Nettoyer les données (par ex. : supprimer les valeurs manquantes).
    3. Encoder les labels catégoriels (colonne `family_accession`) en entiers.
    4. Diviser les données en ensembles d’entraînement, de validation et de test selon une logique définie.
    5. Sauvegarder les ensembles prétraités et des métadonnées utiles.

    Indices :
    - Utilisez `LabelEncoder` pour encoder les catégories.
    - Utilisez `train_test_split` pour diviser les indices des données.
    - Utilisez `to_csv` pour sauvegarder les fichiers prétraités.
    - Calculez les poids de classes en utilisant les comptes des classes.
    """

    # Step 1: Load the data
    print('Loading Data')
    data = pd.read_csv(data_file, index_col=0)

    # Step 2: Handle missing values
    data = data.dropna()

    # Step 3: Encode the 'family_accession' to numeric labels
    label_encoder = LabelEncoder()
    data['class_encoded'] = label_encoder.fit_transform(data['family_accession'])

    # Save the label encoder
    joblib.dump(label_encoder, 'label_encoder.joblib')

    # Save the label mapping to a text file
    with open('label_mapping.txt', 'w', encoding='utf-8') as f:
        for class_label, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
            f.write(f"{class_label} => {encoded_value}\n")

    # Step 4: Distribute data
    # For each unique class:
    # - If count == 1: go to test set
    # - If count == 2: 1 to dev, 1 to test
    # - If count == 3: 1 to train, 1 to dev, 1 to test
    # - Else: stratified split (train/dev/test)

    print("Distributing data")

    train_indices = []
    dev_indices = []
    test_indices = []

    class_counts = data['class_encoded'].value_counts()

    for cls in tqdm.tqdm(class_counts.index):
        cls_indices = data[data['class_encoded'] == cls].index.tolist()
        count = len(cls_indices)

        # Logic for assigning indices to train/dev/test
        if count == 1:
            test_indices.extend(cls_indices)
        
        elif count == 2:
            dev_indices.append(cls_indices[0])
            test_indices.append(cls_indices[1])
        
        elif count == 3:
            train_indices.append(cls_indices[0])
            dev_indices.append(cls_indices[1])
            test_indices.append(cls_indices[2])

        else:
            cls_data = data.loc[cls_indices]
            temp_train, temp_test = train_test_split(cls_data, test_size=0.2, stratify=cls_data['class_encoded'], random_state=42)
            temp_train, temp_dev = train_test_split(temp_train, test_size=0.25, stratify=temp_train['class_encoded'], random_state=42)

            train_indices.extend(temp_train.index.tolist())
            dev_indices.extend(temp_dev.index.tolist())
            test_indices.extend(temp_test.index.tolist())

    # Saving preprocessed datasets

    data.loc[train_indices].to_csv(f"{output_dir}/train.csv")
    data.loc[dev_indices].to_csv(f"{output_dir}/dev.csv")
    data.loc[test_indices].to_csv(f"{output_dir}/test.csv")

    # Compute and save class weights
    class_counts = data.loc[train_indices, 'class_encoded'].value_counts()
    total = class_counts.sum()
    class_weights = {cls : total/count for cls, count in class_counts.items()}

    joblib.dump(class_weights, f"{output_dir}/class_weights.joblib")

    # Step 5: Convert index lists to numpy arrays

    train_indices = np.array(train_indices)
    dev_indices = np.array(dev_indices)
    test_indices = np.array(test_indices)

    # Step 6: Create DataFrames from the selected indices

    train_df = data.loc[train_indices]
    dev_df = data.loc[dev_indices]
    test_df = data.loc[test_indices]

    # Step 7: Drop unused columns: family_id, sequence_name, etc.

    cols_to_drop = ['family_id', 'sequence_name', 'family_accession']
    for df in [train_df, dev_df, test_df]:
        for col in cols_to_drop:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

    # Step 8: Save train/dev/test datasets as CSV
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    dev_df.to_csv(f"{output_dir}/dev.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    # Step 9: Calculate class weights from the training set
    # class_counts = ...
    # class_weights = ...

    # Step 10: Normalize weights and scale

    # Step 11: Save the class weights
    # with open(...)

    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to train CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files")
    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
