import pickle
import sys


file_path = '/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/merges.pkl'

NUM_TO_SHOW = 20


def inspect_bpe_merges(filepath):


    try:
        with open(filepath, 'rb') as f:
            merges_data = pickle.load(f)
    except FileNotFoundError:
        print(f"no file '{filepath}' ")
        sys.exit(1)
    except Exception as e:
        print(f"fail to load file {e}")
        sys.exit(1)



    data_type = type(merges_data)
    print(f"datatype: {data_type}")

    if not hasattr(merges_data, '__len__'):
        print(merges_data)
        return

    data_len = len(merges_data)
    print(f"merge num: {data_len}")

    
    print(str(merges_data)[:500])


if __name__ == "__main__":
    inspect_bpe_merges(file_path)