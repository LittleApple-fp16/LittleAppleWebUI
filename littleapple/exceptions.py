class DatasetTypeError(Exception):
    def __init__(self, dataset_name="未知数据集", message="未知数据集类型"):
        self.dataset_name = dataset_name
        self.message = "数据集类型错误: "+message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.dataset_name} -> {self.message}'
