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
    PARL_AI_ROOT_DIR = "../ParlAI"
    LABEL_STUDIO_ROOT_DIR = "../label-studio"
    BRAT_ROOT_DIR = "../brat"
    HUGGING_FACE_ROOT_DIR = "../hugging_face"
    SECRET_KEY_DIR = "../secret_keys/connection_key.txt"
    LAW_SERVICE_MAPPING_DIR = f"{HUGGING_FACE_ROOT_DIR}/id2label.json"

    class ParlAiPath:
        def __init__(self):
            parlai_root = Directories.PARL_AI_ROOT_DIR
            self.data_dir = f"{parlai_root}/data"
            self.model_dir_root = f"{parlai_root}/models"
            self.log_dir_root = f"{parlai_root}/logs"

    class HuggingFacePath:
        def __init__(self):
            huggingface_root = Directories.HUGGING_FACE_ROOT_DIR
            self.train_model_dir_root = f"{huggingface_root}/models"
            self.eval_model_dir_root = f"{huggingface_root}/models_saved"
            self.log_dir_root = f"{huggingface_root}/logs"
            self.explicitly_given_test_data = "./very_small_test_data.json"
            self.human_inputs_file_dir = "./inputs_for_human_interaction_simulation.json"
            self.evaluation_summary_file_header = "evaluation_summary"
            self.dir_for_label_id2label_text = "./label_id2text.json"
