import json


filename = "/Users/aamirali/Documents/Rabbit/Datasets/COCO/v2_OpenEnded_mscoco_train2014_questions2.json"
data = json.load(filename)

# with open(os.path.join(data_dir, filename), 'r') as f:
#     text_csv = csv.DictReader(f)
#     self._data = []
#     self._label = []
#     for col in text_csv:
#         self._data.append(col["input"])
#         self._label.append(col["output"])