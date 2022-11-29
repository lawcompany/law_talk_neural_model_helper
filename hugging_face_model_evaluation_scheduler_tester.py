class HuggingFaceModelEvaluationSchedulerTester:
    def __init__(self):
        from backend.src.basic_tools.misc import CudaOperators
        op_ = CudaOperators()
        self.whether_to_use_small_samples_for_eval = True
        self.ignore_cache = True
        self.num_gpu = op_.get_the_num_of_available_gpus()
        self.gpu_to_use = 1

    def main(self):
        self._run_auto_eval_in_single_process()
        self._run_summarization_for_evaluation_results()

    @staticmethod
    def _run_summarization_for_evaluation_results():
        from backend.src.hugging_face_model_evaluation_scheduler import HuggingFaceModelEvaluationSummarizer
        summarizer = HuggingFaceModelEvaluationSummarizer()
        summarizer()

    def _run_auto_eval_in_single_process(self):
        check_point_dir_list = self.get_check_point_dir_list_for_evaluation()
        self._run_auto_eval(
            check_point_dir_list=check_point_dir_list,
            gpu_id=self.gpu_to_use
        )

    def _run_auto_eval(self, check_point_dir_list, gpu_id=0):
        """ Returns: an int """
        from backend.src.hugging_face_model_evaluation_scheduler import HuggingFaceModelEvaluationScheduler
        scheduler = HuggingFaceModelEvaluationScheduler()
        print(f"[SCHEDULING] evaluation process at GPU_{gpu_id}"
              f"\n\t\t where `using_small_sample_for_eval` is {self.whether_to_use_small_samples_for_eval}"
              f"\n\t\t and `ignore_cache` is {self.ignore_cache}")
        scheduler(
            target_check_point_dir_list=check_point_dir_list,
            use_small_sample=self.whether_to_use_small_samples_for_eval,
            ignore_cache=self.ignore_cache
        )

    def get_check_point_dir_list_for_evaluation(self):
        from glob import glob
        from backend.src.basic_tools.data_structure.constant import Directories
        from backend.src.hugging_face_model_evaluation_scheduler import HuggingFaceModelEvaluationScheduler
        scheduler = HuggingFaceModelEvaluationScheduler(gpu_id=self.gpu_to_use)
        hf_dir = Directories.HuggingFacePath()
        model_dir_list = glob(hf_dir.eval_model_dir_root + "/model*")
        check_point_dir_list = []
        for model_dir in model_dir_list:
            check_point_dir_list += glob(model_dir + "/checkpoint*")
        to_return = []
        warned_ = set()
        for check_point_dir in check_point_dir_list:
            try:
                scheduler.load_tokenizer_from_model_checkpoint_dir(checkpoint_path=check_point_dir)
                to_return.append(check_point_dir)
            except FileNotFoundError as e:
                model_dir = "/".join(check_point_dir.split("/")[:-1])
                if model_dir in warned_:
                    pass
                else:
                    print(f"[WARNING] {model_dir} - opt file not found: {str(e)}")
                    warned_.add(model_dir)
        return to_return
