import argparse

from data_processing import generate_data_sets
from data_processing import utils

from baseline import baseline

utils.log_timestamp()
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
parser.add_argument("-data_dir", help="Input data location", default="./data")
parser.add_argument("-model_dir", help="Output data file location", default="./model")
args = parser.parse_args()

# Use the project data processing directives
utils.log_timestamp()
training_data = generate_data_sets.read_dataset(name="training",
                                                in_dir=args.data_dir)
validation_data = generate_data_sets.read_dataset(name="validation",
                                                  in_dir=args.data_dir)
test_data = generate_data_sets.read_dataset(name="test",
                                            in_dir=args.data_dir)

utils.log_timestamp()
run_config, hparams = baseline.baseline(args.model_dir)

utils.log_timestamp()
print("Starting training...")
baseline.train(run_config, training_data, validation_data, hparams)
print("Done.")

utils.log_timestamp()
print("\nRunning evaluation...")
evaluation_dict = baseline.test(run_config, validation_data, hparams)
print("Done.")
print(evaluation_dict)
utils.log_timestamp()
