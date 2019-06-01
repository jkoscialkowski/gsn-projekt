"""import modules"""
import math

"""classes"""
class Train_test_split():
    def __init__(self, amReader, input_reg = 'ASIA_PACIFIC', pred_reg = 'EMEIA', int_start = 0, int_end = 10, train_size = 0.8):
        """
        Class Train_test_split

        :param amReader: object from AmphibianReader class
        :param input_reg: region of input observations
        :param pred_reg: region of prices, which we would like to predict
        :param int_start: start of the whole interval
        :param int_end: end of the whole interval
        :param train_size: size of the train set
        """
        self.amReader = amReader
        self.whole_set = {'train_obs': self.amReader.torch[input_reg][int_start:math.floor(int_end * train_size), :, :],
                          'train_y': self.amReader.torch[pred_reg][int_start:math.floor(int_end * train_size), 5, 0],
                          'test_obs': self.amReader.torch[input_reg][math.floor(int_end * train_size):int_end, :, :],
                          'test_y': self.amReader.torch[pred_reg][math.floor(int_end * train_size):int_end, 5, 0]}

    def __getitem__(self, item):
        """
        :param item: one of four: 'train_obs', 'train_y', 'test_obs', 'test_y'
        :return: tensor with values
        """
        return self.whole_set[item]