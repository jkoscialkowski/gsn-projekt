import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from matplotlib.collections import QuadMesh
from torch.utils.data import DataLoader

NO_COMPANIES = 10


class ConfusionMatrix:
    def __init__(self, confmat, class_labels, truth_along_y=True,
                 figsize=(9, 9)):
        self.confmat = confmat
        self.class_labels = class_labels
        self.truth_along_y = truth_along_y
        self.figsize = figsize

    @staticmethod
    def create_new_fig(fn, figsize):
        """ Init graphics """
        fig1 = plt.figure(fn, figsize)
        ax1 = fig1.gca()  # Get current axis
        ax1.cla()  # clear existing plot
        return fig1, ax1

    def compute_metrics(self):
        rowsums, colsums = self.confmat.sum(axis=1), self.confmat.sum(axis=0)

        arr = np.concatenate([
            self.confmat,
            (np.diag(self.confmat) / rowsums).reshape(-1, 1)
        ], axis=1)
        arr = np.concatenate([
            arr,
            np.concatenate([
                (np.diag(self.confmat) / colsums).reshape(1, -1),
                np.asarray(
                    np.diag(self.confmat).sum() / self.confmat.sum()
                ).reshape(1, 1)
            ], axis=1)
        ])
        self.confmat_augmented = arr
        return arr

    @staticmethod
    def get_bg_color(self, pos):
        size = self.confmat.shape[0]
        if pos[0] == pos[1] and pos[0] == size:
            return [0.17, 0.20, 0.17, 1.0]
        elif pos[0] == pos[1]:
            return [0.35, 0.8, 0.55, 1.0]
        elif pos[0] == size or pos[1] == size:
            return [0.27, 0.30, 0.27, 1.0]
        else:
            return [0, 0, 0, 0]

    @staticmethod
    def prepare_texts(self, pos, curr_text):
        size = self.confmat.shape[0]
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum',
                           fontproperties=fm.FontProperties(weight='bold',
                                                            size=20))
        row, col = int(pos[1]), int(pos[0])
        if row == col and row == size:
            return [
                {'x': curr_text._x, 'y': curr_text._y - 0.2, 'text': 'Accuracy',
                 'kw': text_kwargs},
                {'x': curr_text._x, 'y': curr_text._y + 0.2,
                 'text': '{:4.2f}%'.format(
                     self.confmat_augmented[size, size] * 100),
                 'kw': text_kwargs}]
        elif row == col:
            return [{'x': curr_text._x, 'y': curr_text._y,
                     'text': '{:d}'.format(
                         int(self.confmat_augmented[row, col])),
                     'kw': text_kwargs}]
        elif row == size or col == size:
            return [{'x': curr_text._x, 'y': curr_text._y,
                     'text': '{:4.2f}%'.format(
                         self.confmat_augmented[row, col] * 100),
                     'kw': text_kwargs}]
        else:
            text_kwargs['color'] = 'black'
            return [{'x': curr_text._x, 'y': curr_text._y,
                     'text': '{:d}'.format(
                         int(self.confmat_augmented[row, col])),
                     'kw': text_kwargs}]

    def plot(self, fontsize):
        # The first argument is for always plotting in the same window
        fig, ax1 = self.create_new_fig('Conf matrix default', self.figsize)
        ax1.set_title('Confusion matrix', fontsize=fontsize)

        # Add precision, recall and accuracy
        if not hasattr(self, 'confmat_augmented'):
            self.compute_metrics()

        # Create Seaborn heatmap
        ax = sns.heatmap(self.confmat_augmented, annot=True,
                         annot_kws={"fontsize": 20}, linewidths=0.5, ax=ax1,
                         cbar=False, cmap='Oranges', linecolor='w', fmt='.2f')

        # Move xaxis ticks to top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        # Set axis labels and ticks according to confmat orientation
        rec_ticks = self.class_labels + ['Recall']
        prec_ticks = self.class_labels + ['Precision']
        if self.truth_along_y:
            ax.set_xlabel('Predicted', fontsize=fontsize)
            ax.set_ylabel('Actual', fontsize=fontsize)
            ax.set_xticklabels(prec_ticks, rotation=45, fontsize=fontsize)
            ax.set_yticklabels(rec_ticks, rotation=25, fontsize=fontsize)

        else:
            ax.set_xlabel('Actual', fontsize=fontsize)
            ax.set_ylabel('Predicted', fontsize=fontsize)
            ax.set_xticklabels(rec_ticks, rotation=45, fontsize=fontsize)
            ax.set_yticklabels(prec_ticks, rotation=25, fontsize=fontsize)

        plt.tight_layout()  # Set tight layout

        # Turn off all the ticks
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # Creating quadmesh
        quadmesh = ax.findobj(QuadMesh)[0]
        facecolors = quadmesh.get_facecolors()

        # Iterate over text elements
        text_add, text_del = [], []
        pos_ind = -1  # From left to right, bottom to top.
        for text in ax.collections[0].axes.texts:  # ax.texts:
            pos_ind += 1
            pos = np.array(text.get_position()) - [0.5, 0.5]
            # Changing face colors
            facecolors[pos_ind] = self.get_bg_color(self, pos)
            # Appending current text to delete list
            text_del.append(text)
            # Create text which should go into the current cell
            text_add += self.prepare_texts(self, pos, text)

        for t in text_del:
            t.remove()
        for t in text_add:
            ax.text(t['x'], t['y'], t['text'], **t['kw'])

        plt.show()


class MAVI:
    def __init__(self, model, valid_dataset):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.model.eval()
        self.nobs = int(len(valid_dataset) / NO_COMPANIES)
        self.dataloader = DataLoader(valid_dataset,
                                     batch_size=self.nobs,
                                     drop_last=True)
        self.loss_dict = {}
        self.perm_loss_dict = {}

    @staticmethod
    def permute_company(tensor: torch.Tensor, company_no: int):
        shuffle_list = [company_no + i * NO_COMPANIES for i in range(6)]
        shuffled = tensor.clone()
        shuffled[:, :, shuffle_list] = shuffled[torch.randperm(tensor.shape[0]),
                                                torch.randperm(tensor.shape[1]),
                                                shuffle_list]
        return shuffled

    def compute_losses(self):
        with torch.no_grad():
            for company_no, batch in enumerate(self.dataloader):
                self.loss_dict[company_no] = self.loss_fn(
                    self.model(batch['test_obs'].permute(1, 0, 2)),
                    batch['test_y']
                ).item()

                # Compute permutational losses for a given company
                self.perm_loss_dict[company_no] = {}

                for perm_comp in range(NO_COMPANIES):
                    self.perm_loss_dict[company_no][perm_comp] = self.loss_fn(
                        self.model(self.permute_company(
                            batch['test_obs'].permute(1, 0, 2)
                        )),
                        batch['test_y']
                    ).item()

    def plot_mavi(self, company_no):
        pass
