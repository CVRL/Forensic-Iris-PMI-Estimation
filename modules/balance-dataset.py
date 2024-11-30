import os
import pandas as pd
from sklearn.utils import resample

# Define the input and output directories
input_dir = './ds-dis-metadata/'
output_dir = './bal-ds-dis-metadata/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define PMI ranges and corresponding classes
pmi_ranges = [(-1, 24), (25, 48), (49, 72), (73, 96), (97, 120), (121, 144), (145, 168), (169, 192),
              (193, 216), (217, 240), (241, 264), (265, 288), (289, 312), (313, 336), (337, 360),
              (361, 384), (385, 408), (409, 1674)]
classes = list(range(1, 19))


# Function to map PMI to class
def map_pmi_to_class(pmi):
    for i, (lower, upper) in enumerate(pmi_ranges):
        if lower <= pmi <= upper:
            return int(classes[i])
    return None


# Get filenames and filter out unwanted files
filenames = [f for f in os.listdir(input_dir) if f.endswith('.txt') and not f.startswith('.DS_Store')]

for filename in filenames:
    print(f'Processing file: {filename}')

    # Read the data
    df = pd.read_csv(os.path.join(input_dir, filename))
    print(f'Data length: {len(df)}')

    # Assign classes to each row
    df['Class'] = df['pmi'].apply(map_pmi_to_class)

    # Drop rows where class mapping failed
    df = df.dropna(subset=['Class'])
    print(f'Data length after dropping NaNs: {len(df)}')

    # Print the number of samples for each class before balancing
    print("\nNumber of samples per class before balancing:")
    class_counts = df['Class'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} samples")

    # Determine the maximum number of samples in any class
    max_samples_per_class = df['Class'].value_counts().max()
    print(f'\nMaximum number of samples in any class: {max_samples_per_class}\n')

    # Store the balanced dataset in a new DataFrame
    balanced_df = pd.DataFrame()

    # Balance each class by up-sampling with replacement
    for class_label in df['Class'].unique():
        class_samples = df[df['Class'] == class_label]
        balanced_class_samples = resample(class_samples,
                                          replace=True,  # Allow resampling of the same instance
                                          n_samples=max_samples_per_class,
                                          random_state=42)
        balanced_df = pd.concat([balanced_df, balanced_class_samples])

    # Shuffle the balanced DataFrame to mix the samples
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_df = balanced_df.sort_values('pmi')
    print(balanced_df.head(10))

    # Print the number of samples for each class after balancing
    print("\nNumber of samples per class after balancing:")
    balanced_class_counts = balanced_df['Class'].value_counts().sort_index()
    for class_id, count in balanced_class_counts.items():
        print(f"Class {class_id}: {count} samples")

    # Save the balanced DataFrame to a new CSV file
    output_file = os.path.join(output_dir, f'bal-{filename}')
    balanced_df = balanced_df.drop(columns='Class')
    balanced_df.to_csv(output_file, index=False)
    print(f'Saved balanced data to: {output_file}\n')
