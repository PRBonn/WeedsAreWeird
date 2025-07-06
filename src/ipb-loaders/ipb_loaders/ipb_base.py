#!/usr/bin/env python3

import os
import subprocess
from torch.utils.data import Dataset, default_collate


class IPB_Base(Dataset):

    def __init__(self, data_source=None, path_in_dataserver=None, overfit=False, unittesting=False):
        super().__init__()
        assert path_in_dataserver, 'path_in_dataserver must be specified'
        self.overfit = overfit
        self.path_in_dataserver = path_in_dataserver
        self.class_name = self.__class__.__name__
        self.local_user_name = os.environ.get('USER')
        if not data_source:
            self.data_source = self.get_default_path()
            self.check_dataset_exists()
        else:
            self.data_source = data_source
            self.check_dataset_exists()

        if unittesting:
            self.local_user_name = os.environ.get('USER')
            self.check_remote_dir_exists()
        # else:
        #     self.local_user_name = self.set_user_name()

    def download(self):
        cmd = "rsync --recursive --times --compress --delete --progress {}@131.220.233.14:{} {}".format(self.local_user_name, self.path_in_dataserver,
                                                                                                        self.data_source)
        print('\n Downloading dataset from server...')
        subprocess.call(cmd, shell=True)  # TODO how to suppres print on terminal
        print('Done!')
        if os.path.isfile(os.path.join(self.data_source, self.path_in_dataserver.split('/')[-1])):
            self.extract(os.path.join(self.data_source, self.path_in_dataserver.split('/')[-1]))

    def extract(self, file):
        if 'zip' in file:
            cmd = 'unzip {} -d {}'.format(file, self.data_source)
        elif 'gz' in file:
            cmd = 'tar -xvzf {} -C {}'.format(file, self.data_source)
        else:
            raise NotImplemented

        subprocess.call(cmd, shell=True)

    def check_dataset_exists(self):
        exists = os.path.isdir(self.data_source)
        if not exists:
            self.check_user_input(self.data_source)
            self.local_user_name = self.set_user_name()
            os.makedirs(self.data_source)
            self.download()
        else:
            if self.is_empty(self.data_source):
                self.check_user_input(self.data_source)
                self.local_user_name = self.set_user_name()
                self.download()
            else:
                print('Dataset found!')

    def set_user_name(self):
        local_user_name = os.environ.get('USER')
        out_str = """ 
                    We are about to download the dataset from our server. \n
                    Your current username is {} . Press Enter if you want to keep it, otherwise insert remote user name:  
                  """.format(local_user_name)

        value = input(out_str)
        if value:
            return value

        return local_user_name

    def check_remote_dir_exists(self):
        devnull = open(os.devnull, 'w')
        cmd = "ssh {}@131.220.233.14 'ls {}'  ".format(self.local_user_name, self.path_in_dataserver)
        exit_code = subprocess.call(cmd, shell=True, stdout=devnull, stderr=devnull)
        if exit_code > 0:
            raise ValueError('remote folder does not exist')

    def get_default_path(self):
        abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        abs_path = os.path.join(abs_path, 'data')

        return os.path.join(abs_path, self.class_name, self.path_in_dataserver.split("/")[-1])

    def check_user_input(self, path):
        out_str = """
        Downloading the dataset at {} \n
        Are you sure? [y]/n \n 
        """.format(path)

        while True:
            value = input(out_str or 'y')
            value = value.lower()

            yes = {'yes', 'y', ''}
            no = {'no', 'n'}

            if value in yes:
                break
            elif value in no:
                print("Define a different path with {}(data_source=<path>)".format(self.class_name))
                exit(0)
            else:
                print("Please respond with 'y' or 'n'")

    @staticmethod
    def collate(batch):
        return default_collate(batch)

    @staticmethod
    def is_empty(dir_path):
        content = os.listdir(dir_path)
        if content:
            return False
        else:
            return True
