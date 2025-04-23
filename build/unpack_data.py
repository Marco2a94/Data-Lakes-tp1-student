import os
import pandas as pd

def unpack_data(input_dir, output_file):
    """
    Unpacks and combines multiple CSV files from a directory into a single CSV file.

    Parameters:
    input_dir (str): Path to the directory containing the CSV files.
    output_file (str): Path to the output combined CSV file.
    """

    # Step 1: Initialize an empty list to store DataFrames
    dataframes = []

    # Step 2: Loop over files in the input directory
    for filename in os.listdir(input_dir):
        # print("---filename:", filename)
        # Step 3: Check if the file is a CSV or matches a naming pattern
        if filename.endswith(".csv") or filename.startswith("data-"):

            # Step 4: Read the CSV file using pandas
            df = pd.read_csv(os.path.join(input_dir, filename), index_col=0)
            # print("--- head of df:\n", df.head())
            # Step 5: Append the DataFrame to the list
            dataframes.append(df)

    # Step 6: Concatenate all DataFrames
    combined_df = pd.concat(dataframes, axis=0, join="outer")
    # print(f"--- combined df \n{combined_df}")
    # Step 7: Save the combined DataFrame to output_file
    combined_df.to_csv(output_file)
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack and combine protein data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output combined CSV file")
    args = parser.parse_args()

    unpack_data(args.input_dir, args.output_file)
