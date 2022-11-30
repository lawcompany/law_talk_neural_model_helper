class Tokens:
    SUCCESS_CODE = "[SUCCESS]"


class Numbers:
    A_VERY_SMALL_NUMBER = 3
    A_SMALL_NUMBER = 30
    A_MODERATELY_SMALL_NUMBER = 100
    A_MODERATELY_LARGE_NUMBER = 1000
    A_LARGE_NUMBER = 10000
    NUM_LINES_IN_SAMPLE_FILES = 300000


class Directories:
    RESULTS_DIR = "./results"
    LAW_SERVICE_MAPPING_DIR = "./law_ids/label_id2text.json"
    ENV_PARAM_DIR = "./parameters/env_parameters.json"

    class HuggingFacePath:
        def __init__(self):
            self.huggingface_root = Directories.RESULTS_DIR
            self.train_model_dir_root = f"{self.huggingface_root}/models"
            self.eval_model_dir_root = f"{self.huggingface_root}/models"
            self.log_dir_root = f"{self.huggingface_root}/logs"
            self.human_inputs_file_dir = "./inputs_for_human_interaction_simulation.json"
            self.evaluation_summary_file_header = "evaluation_summary"
            self.explicitly_given_test_data = "./dataset_for_sanity_check.json"
