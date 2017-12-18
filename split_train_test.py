import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Split input data for test/train")
parser.add_argument("--data", type=str, dest="data_path", help="Path to the data", required=True)
parser.add_argument("--ratio", type=float, dest="ratio", default=0.33, help="test/train datasets ratio")
parser.add_argument("--balanced", action="store_true", help="make balanced dataset for test")
parser.add_argument("--compressions", type=str, dest="compression", help="type of compression to use")

def load_data(data_path):
    data = pd.read_csv(data_path, delimiter=";", header=None)

    # give name to the columns for better debugging
    feature_column_names = {idx + 2: "f{}".format(idx) for idx in range(len(data.columns) - 2)}
    columns_names = {0: "key", 1: "group"}
    columns_names.update(feature_column_names)
    data.rename(columns=columns_names, inplace = True)

    X = data[list(feature_column_names.values()) + ["key",]]
    y = data["group"]

    return X, y

def get_basename_and_extension(path):
    basename = os.path.basename(path)
    extension = None
    if "." in basename:
        basename, extension = basename.rsplit(".", 1)

    return basename, extension

def build_path(basename, postfix, extension, compression):
    if extension is None:
        name = "{}_{}".format(basename, postfix)
    else:
        name = "{}_{}.{}".format(basename, postfix, extension)

    if compression is not None:
        name = "{}.{}".format(name, compression)

    return name

def main():
    args = parser.parse_args()

    X, y = load_data(args.data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    if args.balanced:
        balanced_data_size = min((y_test == "t").sum(), (y_test == "f").sum())
        X_test_balanced_f = X_test[(y_test=="f")][:balanced_data_size]
        y_test_balanced_f = y_test[(y_test=="f")][:balanced_data_size]

        X_test_balanced_t = X_test[(y_test=="t")][:balanced_data_size]
        y_test_balanced_t = y_test[(y_test=="t")][:balanced_data_size]

        X_test = pd.concat([X_test_balanced_f, X_test_balanced_t])
        y_test = np.concat([y_test_balanced_f, y_test_balanced_t])

    data_test = pd.concat([X_test, y_test], axis=1)
    data_train = pd.concat([X_train, y_train], axis=1)

    basename, extension = get_basename_and_extension(args.data_path)
    train_filename = build_path(basename, "train", extension, compression=args.compression)
    test_filename = build_path(basename, "test", extension, compression=args.compression)
    data_test.to_csv(test_filename, index=False, header=None, sep=";", compression=args.compression)
    data_train.to_csv(train_filename, index=False, header=None, sep=";", compression=args.compression)

if __name__ == '__main__':
    main()

