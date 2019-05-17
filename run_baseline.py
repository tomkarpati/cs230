import sys
from data_processing import generate_data_sets
from baseline import baseline

# Use the project data processing directives
data_dir = sys.argv[1]
model_dir = sys.argv[2]
validation_data = generate_data_sets.read_dataset(name="validation", in_dir=data_dir)
training_data = generate_data_sets.read_dataset(name="training", in_dir=data_dir)
test_data = generate_data_sets.read_dataset(name="test", in_dir=data_dir)

run_config, hparams = baseline.baseline(model_dir)
baseline.train(run_config, training_data, validation_data, hparams)

it = baseline.test(run_config, validation_data, hparams)


  
