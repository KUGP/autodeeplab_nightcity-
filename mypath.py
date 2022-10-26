class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return 'C:/Users/CoIn241/Desktop/citydata'

        elif dataset == 'nightcity':
            return 'C:/Users/CoIn241/Desktop/nightcitydata'

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError