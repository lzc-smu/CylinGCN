class DatasetCatalog(object):
    dataset_attrs = {
        'PAT3DTrain': {
            'id': 'PAT',
            'data_root': 'data/PAT/train/JPEGImages',
            'ann_file': 'data/PAT/train/annotations.json',
            'split': 'train'
        },
        'PAT3DVal': {
            'id': 'PAT',
            'data_root': 'data/PAT/val/JPEGImages',
            'ann_file': 'data/PAT/val/annotations.json',
            'split': 'val'
        },
        'OCT3DTrain': {
            'id': 'OCT',
            'data_root': 'data/OCT/train/JPEGImages',
            'ann_file': 'data/OCT/train/annotations.json',
            'split': 'train'
        },
        'OCT3DVal': {
            'id': 'OCT',
            'data_root': 'data/OCT/val/JPEGImages',
            'ann_file': 'data/OCT/val/annotations.json',
            'split': 'val'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()

