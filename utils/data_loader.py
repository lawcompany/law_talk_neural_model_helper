import json
import jsonlines


class TEXTMerger:
    def __init__(self, head_):
        self.head = head_
        self.files_to_merge = []
        self.merged_text = ""
        self.save_dir = f"{head_}.txt"

    def __call__(self):
        """ Returns: None """
        self.collect_files_to_merge()
        self.merge_collected_files()
        self.save_merged_file()

    def collect_files_to_merge(self):
        """ Returns: None """
        from glob import glob
        dir_list = glob(f"{self.head}*")
        for dir_ in dir_list:
            if dir_ == self.save_dir:
                pass
            else:
                self.files_to_merge.append(dir_)

    def merge_collected_files(self):
        """ Returns: None """
        from tqdm import tqdm
        text_list = []
        print(f"[MERGING] {self.save_dir}")
        for dir_ in tqdm(self.files_to_merge):
            text_ = TEXTLoader(dir_=dir_)()
            text_list.append(text_)
        self.merged_text = "\n".join(text_list)

    def save_merged_file(self):
        """ Returns: None """
        TEXTSaver(dir_=self.save_dir, data_=self.merged_text)()


class TEXTLoader:
    def __init__(self, dir_, delimiter=","):
        self.dir_ = dir_
        self.data = self._load(dir_=dir_, delimiter=delimiter)

    def __call__(self):
        return self.data

    def _load(self, dir_, delimiter=","):
        import codecs
        f_ = codecs.open(dir_, encoding="utf-8")
        str_ = f_.read()
        f_.close()
        return str_

    def get_line_list(self):
        return [line.strip() for line in self.data.split("\n") if line]


class SerializedTEXTLoader(TEXTLoader):
    def __init__(self, dir_, whether_limit_the_number_of_lines_to_load=True, number_of_lines_to_load=10):
        """
        Args:
            dir_: a str
            whether_limit_the_number_of_lines_to_load: a bool
            number_of_lines_to_load: an int
        """
        self.number_of_lines_to_load = number_of_lines_to_load
        self.whether_limit_the_number_of_lines_to_load = whether_limit_the_number_of_lines_to_load
        super().__init__(dir_)

    def _load(self, dir_, delimiter=','):
        sample_lines = []
        with open(dir_, encoding="utf-8") as infile:
            line_cnt = 0
            for line in infile:
                line_cnt += 1
                if (line_cnt > self.number_of_lines_to_load) and self.whether_limit_the_number_of_lines_to_load:
                    break
                else:
                    pass
                sample_lines.append(line)
        str_ = "\n".join(sample_lines)
        return str_

    def collect_return_values_from_serially_called_function_for_the_lines(self, function_that_has_an_argument_of_line):
        """
        Args:
            function_that_has_an_argument_of_line: a function that returns a non-typed obj
                                                   and takes an argument of a str.

        Returns:
            a list of the non-typed obj, where each item is the returned value of the given function
        """
        from tqdm import tqdm
        to_return = []
        print(f"[Opening] {self.dir_}")
        with open(self.dir_, encoding="utf-8") as infile:
            line_cnt = 0
            for line in tqdm(infile):
                line_cnt += 1
                returned_obj = function_that_has_an_argument_of_line(line)
                to_return.append(returned_obj)
        return to_return


class CSVLoader(TEXTLoader):
    def _load(self, dir_, delimiter=','):
        import csv
        data_list = []
        with open(self.dir_, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                data_list.append(row)
        return data_list


class TEXTSaver:
    def __init__(self, data_, dir_, tail_=None):
        self.data_ = data_
        if tail_:
            self.dir_ = self.add_tail(dir_, tail_)
        else:
            self.dir_ = dir_
        self.loader_class = TEXTLoader

    def __call__(self):
        if isinstance(self.data_, str):
            self._backup_original_file_if_override(dir_=self.dir_)
            self._save(data_=self.data_, dir_=self.dir_)
        else:
            raise TypeError("the text saving is not yet implemented for the type that is not str.")

    def _save(self, data_, dir_):
        import codecs
        f_ = codecs.open(dir_, mode='w', encoding='utf8')
        print(data_.strip(), file=f_)
        f_.close()

    def _backup_original_file_if_override(self, dir_):
        """
        before save a file, backup the file, after checking
        whether there is already a file with the same file name,
        in other words, this function backup the original file
        if the function is attempting to override it.

        Args:
            dir_: the dir to check

        Returns: None
        """
        import os
        if os.path.isfile(dir_):
            loader = self.loader_class(dir_)
            data_to_backup = loader()
            tail_index, loop_count, limit_loop_count = 0, 0, 1000
            while True:
                tail_index += 1
                loop_count += 1
                if loop_count > limit_loop_count:
                    break
                backup_dir = self.add_tail(dir_=dir_,
                                           tail_=f".{tail_index}.")
                if os.path.isfile(backup_dir):
                    pass
                else:
                    break
            self._save(
                dir_=backup_dir,
                data_=data_to_backup
            )

    @staticmethod
    def add_tail(dir_, tail_):
        dir_front, suffix = ".".join(dir_.split(".")[:-1]), dir_.split(".")[-1]
        dir_ = dir_front + tail_ + suffix
        return dir_


class JSONSaver(TEXTSaver):
    def __init__(self, data_, dir_, tail_=None):
        super().__init__(data_, dir_, tail_=tail_)
        self.loader_class = JSONLoader

    def __call__(self):
        if isinstance(self.data_, list) or isinstance(self.data_, dict):
            self._backup_original_file_if_override(dir_=self.dir_)
            self._save(data_=self.data_, dir_=self.dir_)
        else:
            raise TypeError("the json saving is not yet implemented for the type that is neither list nor dict.")

    def _save(self, data_, dir_):
        with open(dir_, 'w') as f_:
            json.dump(data_, f_,
                      indent=4, sort_keys=False, ensure_ascii=False)
        return 1


class CSVSaver(TEXTSaver):
    def __init__(self, data_, dir_, header=None, tail_=None):
        super().__init__(data_, dir_, tail_=tail_)
        self.header = header
        self.loader_class = CSVLoader

    def __call__(self):
        if isinstance(self.data_, list):
            self._backup_original_file_if_override(dir_=self.dir_)
            self._save(data_=self.data_, dir_=self.dir_)
        else:
            raise TypeError("the csv saving is not yet implemented for the type that is not list.")

    def _save(self, data_, dir_):
        import pandas as pd
        data_frames = pd.DataFrame(data_)
        if self.header:
            data_frames.to_csv(dir_, header=self.header, index=False)
        else:
            data_frames.to_csv(dir_, header=False, index=False)


class JSONLoader(TEXTLoader):
    def _load(self, dir_, delimiter=','):
        """
        Args:
            dir_: the dir of the json file

        Returns: dict (if the target object is a dict) or
                 a list of dict (if the target object is a list of dict - id est, jsonl)
        """
        with open(dir_) as json_file:
            loaded_obj = json.load(json_file)
        if isinstance(loaded_obj, dict):
            return loaded_obj
        assert isinstance(loaded_obj, list), (
            f"the type of the loaded object from json should be list if it is not a dict, but it is {type(loaded_obj)}."
        )
        if len(loaded_obj) == 1:
            loaded_obj = loaded_obj[0]
            assert isinstance(loaded_obj, dict), (
                f"the type should be dict if the len of the object is 1, but it is {type(loaded_obj)}."
            )
        return loaded_obj


class JSONLLoader(TEXTLoader):
    def _load(self, dir_, delimiter=''):
        """
        :param dir_: the dir of the jsonl file
        :return:
        """
        import json
        dict_list = []
        with open(dir_, 'r') as json_file:
            json_list = list(json_file)
            for json_str in json_list:
                dict_ = json.loads(json_str)
                dict_list.append(dict_)
        return dict_list


class FileCollector:
    @staticmethod
    def recursively_collect_data_directories(dir_root, tail):
        """
        This method collect all the file names under the directory, recursively.

        Args:
            dir_root: a str
            tail: a str, e.g., 'txt' or 'jsonl'

        Returns:
            a list of str
        """
        from pathlib import Path
        dir_list = []
        for dir_ in Path(dir_root).rglob(f'*.{tail}'):
            dir_ = str(dir_)
            dir_list.append(dir_)
        return dir_list


class KorAsciiConverter:
    @staticmethod
    def convert_jsonlines_to_non_ascii(read_dir, write_dir):
        """
        load the data at the read_dir and convert the data into non-ascii format
        then save it into the write_dir. it can be necessary
        when the Korean data should be seen in the terminal, directly.

        Args:
            read_dir: a str
            write_dir: a str

        Returns:
            None
        """
        dict_list = []
        with jsonlines.open(read_dir) as read_file:
            for line in read_file.iter():
                dict_list.append(line)
        out_lines = []
        for dict_ in dict_list:
            out_lines.append(json.dumps(dict_, ensure_ascii=False))
        with open(write_dir, mode='w', encoding='utf-8') as write_file:
            out_lines = "\n".join(out_lines)
            write_file.write(out_lines)


class NestedDirectoryCreator:
    @staticmethod
    def create_nested_directory_for_the_depth_of_two(root_dir, dir_list_under_the_root):
        """
        Args:
            root_dir: a str
            dir_list_under_the_root: a list of str
        """
        import os
        if os.path.isdir(root_dir):
            pass
        else:
            print(f"[MAKING DIR] {root_dir}")
            os.mkdir(root_dir)
        for dir_ in dir_list_under_the_root:
            if os.path.isdir(dir_):
                pass
            else:
                print(f"[MAKING DIR] {dir_}")
                os.mkdir(dir_)

