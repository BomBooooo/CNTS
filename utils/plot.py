from symbol import factor
import numpy as np
import matplotlib.pyplot as plt


class PlotResult():
    def __init__(self, logger_path, args):
        self.args = args
        self.logger_path = logger_path
        self.color_tuple = {
            0: 'black', 
            1: 'green', 
            2: 'red', 
            3: 'blue', 
            4: 'orange', 
            5: 'purple', 
            6: 'yellow'
        }

    def plot_solo_d(self, itr, epoch, plot_dict, loss, name, task):

        det_output = np.array(plot_dict['score'])
        labels = np.array(plot_dict['label'])
        imp_input = np.array(plot_dict['imp_input'])
        imp = np.array(plot_dict['imp'])
        loss_mask = np.array(plot_dict['loss_mask'])
        batch_x = np.array(plot_dict['x'])
        batch_y = np.array(plot_dict['y'])

        loss_ind = np.where(loss_mask > 0)[0]
        det_out_ind = np.where(det_output > plot_dict['threshold'])[0]
        ab_ind = np.where(batch_y > 0)[0]

        x = np.arange(batch_x.shape[0])
        fig = plt.figure(figsize=(45,9))

        plt.plot(x, labels, c=self.color_tuple[2], linestyle='--', alpha=0.5, label='label')

        plt.plot(x, batch_x, c=self.color_tuple[1])
        plt.plot(x, imp, c=self.color_tuple[2])
        plt.fill_between(x, batch_x, imp, facecolor=self.color_tuple[3], alpha=0.5)

        plt.scatter(x[ab_ind], batch_x[ab_ind], c=self.color_tuple[0], s=60, label='abnormal')
        plt.scatter(x[det_out_ind], batch_x[det_out_ind], c=self.color_tuple[6], s=20, label='det point')

        plt.legend()
        plt.savefig(self.logger_path + '/%s_idx%s_itr_%s_epo_%s_loss_%.4f.png' % (
            task, self.args.data_name[self.args.idx].split('.')[0], itr+1, epoch+1, loss))
        plt.close()

    def plot_solo_i(self, itr, epoch, plot_dict, loss, name, task):
    
        batch_x = np.array(plot_dict['x'])
        batch_y = np.array(plot_dict['y'])
        imp = np.array(plot_dict['imp'])
        imp_index = np.array(plot_dict['imp_index'])
        imp_input = np.array(plot_dict['imp_input'])

        imp_ind = np.where(imp_index > 0)[0]
        ab_ind = np.where(batch_y > 0)[0]

        x = np.arange(len(batch_x))
        fig = plt.figure(figsize=(45,9))

        plt.plot(x, batch_x, c=self.color_tuple[1])
        plt.plot(x, imp, c=self.color_tuple[2])
        plt.fill_between(x, batch_x, imp, facecolor=self.color_tuple[3], alpha=0.5)

        plt.scatter(x[ab_ind], batch_x[ab_ind], c=self.color_tuple[0], s=60, label='abnormal')
        plt.scatter(x[imp_ind], batch_x[imp_ind], c=self.color_tuple[1], s=20, label='miss point')
        plt.legend()

        plt.savefig(self.logger_path + '/%s_idx%s_itr_%s_epo_%s_loss_%.4f.png' % (
            task, self.args.data_name[self.args.idx].split('.')[0], itr+1, epoch+1, loss))
        plt.close()

