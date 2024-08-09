from pathlib import Path

class dataset(object):
    
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

        self.s3_bucket = False
        self.image_present = False
        self.labels_gt = False


    def transform(self):
        pass
    
    def __getitem__(self, index) -> None:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")


    def __len__(self):
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")