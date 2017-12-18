from .testml import do_train, do_test, do_eval
import logging
import argparse

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

launch_modes = ["train", "test", "eval"]

parser = argparse.ArgumentParser(description="Train/test/eval models")
parser.add_argument('--mode', type=str, dest="mode", help="Launch mode: {}".format(launch_modes), required=True)
parser.add_argument("--model", type=str,dest="model_path", help="Path to the model", required=True)
parser.add_argument("--data", type=str, dest="data_path", help="Path to the data", required=True)
parser.add_argument("--eval", type=str, dest="eval_path", help="Path where the evaluation results will be stored. If not set, eval result will be printed to stdout")

def main():
    args = parser.parse_args()
    mode = args.mode

    if mode not in launch_modes:
        logging.error("Unknown mode {}".format(mode))
        return

    if mode == "train":
        do_train(args.data_path, args.model_path)

    if mode == "test":
        do_test(args.data_path, args.model_path)

    if mode == "eval":
        do_eval(args.data_path, args.model_path, args.eval_path)

if __name__ == '__main__':
    main()
