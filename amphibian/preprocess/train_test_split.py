import math


class TrainTestSplit:
    def __init__(self, am_reader, input_reg: str = 'ASIA_PACIFIC', pred_reg: str = 'EMEIA',
                 int_start: int = 0, int_end: int = 10, train_size: float = 0.8):
        """
        Class Train_test_split

        :param am_reader: object from AmphibianReader class
        :param input_reg: region of input observations
        :param pred_reg: region of prices, which we would like to predict
        :param int_start: start of the whole interval
        :param int_end: end of the whole interval
        :param train_size: size of the train set
        """
        int_e = math.floor((int_end - int_start) * train_size + int_start)
        self.am_reader = am_reader
        self.whole_set = {
            'train_obs': self.am_reader.torch[input_reg][int_start:int_e, :, :],
            'train_y': self.am_reader.torch[pred_reg][int_start:int_e, 5, :],
            'test_obs': self.am_reader.torch[input_reg][int_e:int_end, :, :],
            'test_y': self.am_reader.torch[pred_reg][int_e:int_end, 5, :]
        }

    def __getitem__(self, item):
        """
        :param item: one of four: 'train_obs', 'train_y', 'test_obs', 'test_y'
        :return: tensor with values
        """
        return self.whole_set[item]
