import argparse

from data_processing import generate_data_sets
from baseline import baseline

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
parser.add_argument("-data_dir", help="Input data location", default="./data")
parser.add_argument("-model_dir", help="Output data file location", default="./model")
args = parser.parse_args()

# Use the project data processing directives
validation_data = generate_data_sets.read_dataset(name="validation",
                                                  in_dir=args.data_dir)
#training_data = generate_data_sets.read_dataset(name="training",
#                                                in_dir=args.data_dir)
test_data = generate_data_sets.read_dataset(name="test",
                                            in_dir=args.data_dir)

run_config, hparams = baseline.baseline(args.model_dir)
#baseline.train(run_config, training_data, validation_data, hparams)

evaluation_dict = baseline.test(run_config, validation_data, hparams)
print("\nDone.")
print(evaluation_dict)
