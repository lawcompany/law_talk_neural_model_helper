class HuggingFaceHelper:
    def __init__(self, gpu_id=0):
        from os import environ
        from utils.constant import Directories
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.hf_path = Directories.HuggingFacePath()
        self.hf_path_root = self.hf_path.huggingface_root
        self.train_model_dir_root = self.hf_path.train_model_dir_root
        self.eval_model_dir_root = self.hf_path.eval_model_dir_root
        self.log_dir_root = self.hf_path.log_dir_root
        self.create_necessary_directories_if_needed()

    def create_necessary_directories_if_needed(self):
        for path_ in [self.hf_path_root, self.log_dir_root,
                      self.train_model_dir_root, self.eval_model_dir_root]:
            self.create_dir(path_=path_)

    @staticmethod
    def create_dir(path_):
        """
        Args:
            path_: a str

        Returns:
            None
        """
        import os
        if os.path.isdir(path_):
            pass
        else:
            os.mkdir(path_)
