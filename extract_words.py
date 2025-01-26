import pandas as pd
import pickle
import os

def extract_and_save_words(csv_file_path, output_pkl_path):
    try:
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"File not found: {csv_file_path}")

        df = pd.read_csv(csv_file_path)
        columns = df.columns.tolist()

        if len(columns) < 3:
            raise ValueError("Dataset does not have enough columns.")

        word_columns = columns[1:-1]
        expected_num_words = 3000
        actual_num_words = len(word_columns)
        
        if actual_num_words != expected_num_words:
            print(f"Warning: Expected {expected_num_words} words, but extracted {actual_num_words} words.")

        print(f"Number of word columns extracted: {actual_num_words}")
        print("First 10 words:", word_columns[:10])

        with open(output_pkl_path, 'wb') as f:
            pickle.dump(word_columns, f)

        print(f"Word list successfully saved to '{output_pkl_path}'")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    csv_file = 'emails.csv'
    pkl_file = 'words.pkl'
    extract_and_save_words(csv_file, pkl_file)