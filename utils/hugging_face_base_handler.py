class HuggingFaceBaseHandler:
    def __init__(self, gpu_id=0):
        from os import environ
        from utils.constant import Directories
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.hf_path = Directories.HuggingFacePath()
        self.train_model_dir_root = self.hf_path.train_model_dir_root
        self.eval_model_dir_root = self.hf_path.eval_model_dir_root
        self.log_dir_root = self.hf_path.log_dir_root
