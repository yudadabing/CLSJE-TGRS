from operator import index
from simplecv import dp_train as train
import torch
from simplecv.util.logger import eval_progress, speed
import time
from module import CLSJE
from simplecv.util import metric
from simplecv.util import registry
from torch.utils.data.dataloader import DataLoader
from simplecv import registry
from simplecv.core.config import AttrDict
from scipy.io import loadmat,savemat
import data.dataloader
import matplotlib.pyplot as plt
import numpy as np

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list,imageID):
    #ID=1:Pavia University
    #ID=2:Salinas
    # ID=3:     Indian Pines 
    #ID=4:Houston
    #ID=5: KSC
    # ID=6:Botswana
    #ID=7：PaviaC
    #ID=8： SalinasA
    y = np.zeros((x_list.shape[0], 3))
    if imageID ==1:
        for index, item in enumerate(x_list):
            if item == 0:
                y[index] = np.array([216,191,216]) / 255.
            if item == 1:
                y[index] = np.array([0, 255, 0]) / 255.
            if item == 2:
                y[index] = np.array([0, 255, 255]) / 255.
            if item == 3:
                y[index] = np.array([45,138,86]) / 255.
            if item == 4:
                y[index] = np.array([255, 0, 255]) / 255.
            if item == 5:
                y[index] = np.array([255, 165, 0]) / 255.
            if item == 6:
                y[index] = np.array([159, 31, 239]) / 255.
            if item == 7:
                y[index] = np.array([255, 0, 0]) / 255.
            if item == 8:
                y[index] = np.array([255, 255, 0]) / 255.
            if item == -1:
                y[index] = np.array([0, 0, 0]) / 255.
    elif imageID ==2:
         for index, item in enumerate(x_list):
            if item == 0:
                y[index] = np.array([255,0,0]) / 255.
            if item == 1:
                y[index] = np.array([0, 255, 0]) / 255.
            if item == 2:
                y[index] = np.array([0, 0, 255]) / 255.
            if item == 3:
                y[index] = np.array([255,255,0]) / 255.
            if item == 4:
                y[index] = np.array([0, 255, 255]) / 255.
            if item ==5:
                y[index]=np.array([255,0,  255]) / 255.
            if item == 6:
                y[index] = np.array([176, 48,96]) / 255.
            if item == 7:
                y[index] = np.array([46,139,87]) / 255.
            if item == 8:
                y[index] = np.array([160,32,240]) / 255.
            if item == 9:
                y[index] = np.array([255, 127,80]) / 255.
            if item ==10:
                y[index] = np.array([127,255,212]) / 255.
            if item ==11:
                y[index] = np.array( [218,112,214]) / 255.  
            if item ==12:
                y[index] = np.array( [160,82,45]) / 255.  
            if item ==13:
                y[index] = np.array([127,255,0]) / 255.  
            if item ==14:
                y[index] = np.array([216,191,216]) / 255.  
            if item ==15:
                y[index] = np.array( [238,0,0]) / 255.  
            if item == -1:
                y[index] = np.array([0, 0, 0]) / 255.
    elif imageID ==3:
         for index, item in enumerate(x_list):
            if item == 0:
                y[index] = np.array([255,0,0]) / 255.
            if item == 1:
                y[index] = np.array([0, 255, 0]) / 255.
            if item == 2:
                y[index] = np.array([0, 0, 255]) / 255.
            if item == 3:
                y[index] = np.array([255,255,0]) / 255.
            if item == 4:
                y[index] = np.array([0, 255, 255]) / 255.
            if item ==5:
                y[index]=np.array([255,0,  255]) / 255.
            if item == 6:
                y[index] = np.array([176, 48,96]) / 255.
            if item == 7:
                y[index] = np.array([46,139,87]) / 255.
            if item == 8:
                y[index] = np.array([160,32,240]) / 255.
            if item == 9:
                y[index] = np.array([255, 127,80]) / 255.
            if item ==10:
                y[index] = np.array([127,255,212]) / 255.
            if item ==11:
                y[index] = np.array( [218,112,214]) / 255.  
            if item ==12:
                y[index] = np.array( [160,82,45]) / 255.  
            if item ==13:
                y[index] = np.array([127,255,0]) / 255.  
            if item ==14:
                y[index] = np.array([216,191,216]) / 255.  
            if item ==15:
                y[index] = np.array( [238,0,0]) / 255.  
            if item == -1:
                y[index] = np.array([0, 0, 0]) / 255.
    elif imageID ==4:
         for index, item in enumerate(x_list):
            if item == 0:
                y[index] = np.array([0,205,0]) / 255.
            if item == 1:
                y[index] = np.array([127, 255, 0]) / 255.
            if item == 2:
                y[index] = np.array([46, 139, 87]) / 255.
            if item == 3:
                y[index] = np.array([0,139,0]) / 255.
            if item == 4:
                y[index] = np.array([160, 82, 45]) / 255.
            if item ==5:
                y[index]=np.array([0,255,  255]) / 255.
            if item == 6:
                y[index] = np.array([155, 55,85]) / 255.###############
            if item == 7:
                y[index] = np.array([216,191,216]) / 255.
            if item == 8:
                y[index] = np.array([255,0,0]) / 255.
            if item == 9:
                y[index] = np.array([139, 0,0]) / 255.
            if item ==10:
                y[index] = np.array([139, 67, 45]) / 255.
            if item ==11:
                y[index] = np.array( [255,255,0]) / 255.  
            if item ==12:
                y[index] = np.array( [238,154,0]) / 255.  
            if item ==13:
                y[index] = np.array([85,26,139]) / 255.  
            if item ==14:
                y[index] = np.array([255,127,80]) / 255.  
            if item == -1:
                y[index] = np.array([0, 0, 0]) / 255.
    elif imageID ==5:
         for index, item in enumerate(x_list):
            if item == 0:
                y[index] = np.array([0,205,0]) / 255.
            if item == 1:
                y[index] = np.array([127, 255, 0]) / 255.
            if item == 2:
                y[index] = np.array([46, 139, 87]) / 255.
            if item == 3:
                y[index] = np.array([0,139,0]) / 255.
            if item == 4:
                y[index] = np.array([160, 82, 45]) / 255.
            if item ==5:
                y[index]=np.array([0,255,  255]) / 255.
            if item == 6:
                y[index] = np.array([255, 255,255]) / 255.
            if item == 7:
                y[index] = np.array([216,191,216]) / 255.
            if item == 8:
                y[index] = np.array([255,0,0]) / 255.
            if item == 9:
                y[index] = np.array([139, 0,0]) / 255.
            if item ==10:
                y[index] = np.array([139, 67, 45]) / 255.
            if item ==11:
                y[index] = np.array( [255,255,0]) / 255.  
            if item ==12:
                y[index] = np.array( [238,154,0]) / 255.  
            if item == -1:
                y[index] = np.array([0, 0, 0]) / 255.
    elif imageID ==6:
         for index, item in enumerate(x_list):
            if item == 0:
                y[index] = np.array([255,0,0]) / 255.
            if item == 1:
                y[index] = np.array([0, 255, 0]) / 255.
            if item == 2:
                y[index] = np.array([0, 0, 255]) / 255.
            if item == 3:
                y[index] = np.array([255,255,0]) / 255.
            if item == 4:
                y[index] = np.array([0, 255, 255]) / 255.
            if item ==5:
                y[index]=np.array([255,0,  255]) / 255.
            if item == 6:
                y[index] = np.array([176, 48,96]) / 255.
            if item == 7:
                y[index] = np.array([46,139,87]) / 255.
            if item == 8:
                y[index] = np.array([160,32,240]) / 255.
            if item == 9:
                y[index] = np.array([255, 127,80]) / 255.
            if item ==10:
                y[index] = np.array([127,255,212]) / 255.
            if item ==11:
                y[index] = np.array( [218,112,214]) / 255.  
            if item ==12:
                y[index] = np.array( [160,82,45]) / 255.  
            if item ==13:
                y[index] = np.array([127,255,0]) / 255.  
            if item == -1:
                y[index] = np.array([0, 0, 0]) / 255.
    elif imageID ==7:
        for index, item in enumerate(x_list):
            if item == 0:
                y[index] = np.array([216,191,216]) / 255.
            if item == 1:
                y[index] = np.array([0, 255, 0]) / 255.
            if item == 2:
                y[index] = np.array([0, 255, 255]) / 255.
            if item == 3:
                y[index] = np.array([45,138,86]) / 255.
            if item == 4:
                y[index] = np.array([255, 0, 255]) / 255.
            if item == 5:
                y[index] = np.array([255, 165, 0]) / 255.
            if item == 6:
                y[index] = np.array([159, 31, 239]) / 255.
            if item == 7:
                y[index] = np.array([255, 0, 0]) / 255.
            if item == 8:
                y[index] = np.array([255, 255, 0]) / 255.
            if item == -1:
                y[index] = np.array([0, 0, 0]) / 255.
    elif imageID ==8:
         for index, item in enumerate(x_list):
            if item == 0:
                y[index] = np.array([255,0,0]) / 255.
            if item == 1:
                y[index] = np.array([255, 127,80]) / 255.
            if item ==2:
                y[index] = np.array([127,255,212]) / 255.
            if item ==3:
                y[index] = np.array( [218,112,214]) / 255.  
            if item ==4:
                y[index] = np.array( [160,82,45]) / 255.  
            if item ==5:
                y[index] = np.array([127,255,0]) / 255.  
            if item == -1:
                y[index] = np.array([0, 0, 0]) / 255.
    return y



def fcn_evaluate_fn(self, test_dataloader, config):
    if self.checkpoint.global_step < 0:
        return
    self._model.eval()
    #print(self.model_dir[6:len(self.model_dir)-17])
    total_time = 0.
    y_all_list = []
    y_all_gt = []
    with torch.no_grad():
        for idx, (im, mask, w) in enumerate(test_dataloader):
            start = time.time()
            y_pred = self._model(im).squeeze()
            torch.cuda.synchronize()
            time_cost = round(time.time() - start, 3)
            y_pred = y_pred.argmax(dim=0).cpu() + 1
            #print(y_pred.size())
            w.unsqueeze_(dim=0)
            # if  self.model_dir[6:len(self.model_dir)-15] =='pavia':
            #     y_out = y_pred[0:610, 0:340]
            #     gt_mat = loadmat('./pavia/PaviaU_gt.mat')
            #     gt_mask = gt_mat['paviaU_gt']
            #     ind=1
            # elif  self.model_dir[6:len(self.model_dir)-15] =='salinas':
            #     y_out = y_pred[0:512, 0:217]
            #     gt_mat = loadmat('./salinas/Salinas_gt.mat')
            #     gt_mask = gt_mat['salinas_gt']
            #     ind=2
            # elif  self.model_dir[6:len(self.model_dir)-15] =='Indianpine':
            #     y_out = y_pred[0:145, 0:145]
            #     gt_mat = loadmat('./IndianPines/Indian_pines_gt.mat')
            #     gt_mask = gt_mat['indian_pines_gt']
            #     ind=3
            #  self.model_dir[6:len(self.model_dir)-15] =='HOS':
            y_out = y_pred[0:349, 0:1905]
            gt_mat = loadmat('./GRSS2013/GRSS2013.mat')
            gt_mask = gt_mat['name']
            #     ind=4
            # elif  self.model_dir[6:len(self.model_dir)-15] =='KSC':
            #     y_out = y_pred[0:512, 0:614]
            #     gt_mat = loadmat('./ksc/KSC_gt.mat')
            #     gt_mask = gt_mat['KSC_gt']
            #     ind=5
            # elif  self.model_dir[6:len(self.model_dir)-15] =='Botswana':
            #     y_out = y_pred[0:1476, 0:256]
            #     gt_mat = loadmat('./Botswana/Botswana_gt.mat')
            #     gt_mask = gt_mat['Botswana_gt']
            #     ind=6 
            # elif  self.model_dir[6:len(self.model_dir)-15] =='paviaC':
            #     y_out = y_pred[0:1096, 0:715]
            #     gt_mat = loadmat('./paviaC/Pavia_gt.mat')
            #     gt_mask = gt_mat['pavia_gt']
            #     ind=7
            # elif  self.model_dir[6:len(self.model_dir)-15] =='salinasA':
            #     y_out = y_pred[0:83, 0:86]
            #     gt_mat = loadmat('./salinasA/SalinasA_gt.mat')
            #     gt_mask = gt_mat['salinasA_gt']
            #     ind=8           
            
            w = w.byte()

            mask = torch.masked_select(mask.view(-1), w.view(-1))
            #print(mask.size())

            y_pred = torch.masked_select(y_pred.view(-1), w.view(-1))
            #print(y_pred.size()          

            gt = gt_mask.flatten()
            x_label = np.zeros(gt.shape)
            y_label = np.zeros(gt.shape)
            for i in range(len(gt)):
                if gt[i] == 0:
                    gt[i] = 17
                    x_label[i] = 16

            gt = gt[:] - 1
            y_out = y_out.flatten()
            for i in range(len(y_out)):
                if y_out[i] == 0:
                    y_out[i] = 17
                    y_label[i] = 16
            y_out = y_out[:] - 1
            x = np.ravel(y_out)
            x1=np.reshape(x,((gt_mask.shape[0], gt_mask.shape[1])))
            x1=x1+1
            ind=4
            if ind ==1:
                savemat('./predict_mat/1.0_poly/pavia_pre.mat',{"pavia_pre":x1})
                y_list = list_to_colormap(x,1)
                y_list2 = list_to_colormap(x,1)
                y_gt = list_to_colormap(gt,1)
            elif ind ==2:
                savemat('./predict_mat/1.0_poly/salinas_pre.mat',{"salinas_pre":x1})
                y_list = list_to_colormap(x,2)
                y_list2 = list_to_colormap(x,2)
                y_gt = list_to_colormap(gt,2)    
            elif ind ==3:
                savemat('./predict_mat/1.0_poly/indianpine_pre.mat',{"indianpine_pre":x1})
                y_list = list_to_colormap(x,3)
                y_list2 = list_to_colormap(x,3)
                y_gt = list_to_colormap(gt,3)
            elif ind ==4:
                savemat('./predict_mat/1.0_poly/hos_pre.mat',{"hos_pre":x1})
                y_list = list_to_colormap(x,4)
                y_list2 = list_to_colormap(x,4)
                y_gt = list_to_colormap(gt,4)
            elif ind ==5:
                savemat('./predict_mat/1.0_poly/KSC_pre.mat',{"KSC_pre":x1})
                y_list = list_to_colormap(x,5)
                y_list2 = list_to_colormap(x,5)
                y_gt = list_to_colormap(gt,5)
            elif ind ==6:
                savemat('./predict_mat/1.0_poly/Botswana_pre.mat',{"Botswana_pre":x1})
                y_list = list_to_colormap(x,6)
                y_list2 = list_to_colormap(x,6)
                y_gt = list_to_colormap(gt,6)
            elif ind ==7:
                savemat('./predict_mat/1.0_poly/paviaC_pre.mat',{"paviaC_pre":x1})
                y_list = list_to_colormap(x,7)
                y_list2 = list_to_colormap(x,7)
                y_gt = list_to_colormap(gt,7)
            elif ind ==8:
                savemat('./predict_mat/1.0_poly/salinasA_pre.mat',{"salinasA_pre":x1})
                y_list = list_to_colormap(x,8)
                y_list2 = list_to_colormap(x,8)
                y_gt = list_to_colormap(gt,8)
            
            y_all_list.append(y_list)
            y_all_gt.append(y_gt)
            y_re = np.reshape(y_list, (gt_mask.shape[0], gt_mask.shape[1], 3))
            y_re_gt = np.reshape(y_list2, (gt_mask.shape[0], gt_mask.shape[1], 3))
            ins=np.where(gt_mask==0)
            y_re_gt[ins]=0
           
            gt_re = np.reshape(y_gt, (gt_mask.shape[0], gt_mask.shape[1], 3))
              
            oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
            #print(oa.numpy())
            aa, acc_per_class = metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                 self._model.module.config.num_classes,
                                                                 return_accuracys=True)
            kappa = metric.th_cohen_kappa_score(mask.view(-1), y_pred.view(-1), self._model.module.config.num_classes)
            total_time += time_cost
            speed(self._logger, time_cost, 'im')

            eval_progress(self._logger, idx + 1, len(test_dataloader))


            if ind ==1:
                classification_map(y_re, gt_mask, 300,
                               './classification_maps/pavia/1.0_poly/' + str(oa.numpy())+ '_' + 'pavia.png')
                classification_map(gt_re, gt_mask, 300,
                               './classification_maps/pavia/1.0_poly/'+ 'pavia_gt.png')
                classification_map(y_re_gt, gt_mask, 300,
                               './classification_maps/pavia/1.0_poly/' + str(oa.numpy())+ '_' + 'pavia_pregt.png')
            elif ind ==2:
                classification_map(y_re, gt_mask, 300,
                               './classification_maps/salinas/1.0_poly/' +str(oa.numpy()) + '_' + 'salinas.png')
                classification_map(gt_re, gt_mask, 300,
                               './classification_maps/salinas/1.0_poly/' + 'salinas_gt.png')
                classification_map(y_re_gt, gt_mask, 300,
                               './classification_maps/salinas/1.0_poly/'+str(oa.numpy()) + 'salinas_pregt.png')
            elif ind ==3:
                classification_map(y_re, gt_mask, 300,
                               './classification_maps/indianpines/1.0_poly/' + str(oa.numpy()) + '_' + 'indianpines.png')
                classification_map(gt_re, gt_mask, 300,
                               './classification_maps/indianpines/1.0_poly/' +   'indianpines_gt.png')
                classification_map(y_re_gt, gt_mask, 300,
                               './classification_maps/indianpines/1.0_poly/' + str(oa.numpy()) + '_' + 'indianpines_pregt.png')
            elif ind ==4:
                classification_map(y_re, gt_mask, 300,
                               './classification_maps/hos/' + str(oa.numpy()) + '_' + 'hos.png')
                classification_map(gt_re, gt_mask, 300,
                               './classification_maps/hos/'  + 'hos_gt.png')
                classification_map(y_re_gt, gt_mask, 300,
                               './classification_maps/hos/' + str(oa.numpy()) + '_' + 'hos_pregt.png')
            elif ind ==5:
                classification_map(y_re, gt_mask, 300,
                               './classification_maps/ksc/1.0_poly/' + str(oa.numpy()) + '_' + 'KSC.png')
                classification_map(gt_re, gt_mask, 300,
                               './classification_maps/ksc/1.0_poly/'  + 'KSC_gt.png')
                classification_map(y_re_gt, gt_mask, 300,
                               './classification_maps/ksc/1.0_poly/' + str(oa.numpy()) + '_' + 'KSC_pregt.png')
            elif ind ==6:
                classification_map(y_re, gt_mask, 300,
                               './classification_maps/Botswana/1.0_poly/' + str(oa.numpy()) + '_' + 'Botswana.png')
                classification_map(gt_re, gt_mask, 300,
                               './classification_maps/Botswana/1.0_poly/'  + 'Botswana_gt.png')
                classification_map(y_re_gt, gt_mask, 300,
                               './classification_maps/Botswana/1.0_poly/' + str(oa.numpy()) + '_' + 'Botswana_pregt.png')
            elif ind ==7:
                classification_map(y_re, gt_mask, 300,
                               './classification_maps/paviaC/1.0_poly/' + str(oa.numpy()) + '_' + 'paviaC.png')
                classification_map(gt_re, gt_mask, 300,
                               './classification_maps/paviaC/1.0_poly/'  + 'paviaC_gt.png')
                classification_map(y_re_gt, gt_mask, 300,
                               './classification_maps/paviaC/1.0_poly/' + str(oa.numpy()) + '_' + 'paviaC_pregt.png')
            elif ind ==8:
                classification_map(y_re, gt_mask, 300,
                               './classification_maps/salinasA/1.0_poly/' + str(oa.numpy()) + '_' + 'salinasA.png')
                classification_map(gt_re, gt_mask, 300,
                               './classification_maps/salinasA/1.0_poly/'  + 'salinasA_gt.png')
                classification_map(y_re_gt, gt_mask, 300,
                               './classification_maps/salinasA/1.0_poly/' + str(oa.numpy()) + '_' + 'salinasA_pregt.png')


    speed(self._logger, round(total_time / len(test_dataloader), 3), 'batched im (avg)')

    metric_dict = {
        'OA': oa.item(),
        'AA': aa.item(),
        'Kappa': kappa.item()
    }
    for i, acc in enumerate(acc_per_class):
        metric_dict['acc_{}'.format(i + 1)] = acc.item()
    self._logger.eval_log(metric_dict=metric_dict, step=self.checkpoint.global_step)


def register_evaluate_fn(launcher):
    launcher.override_evaluate(fcn_evaluate_fn)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = train.parser.parse_args()
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    train.run(config_path=args.config_path,
              model_dir=args.model_dir,
              cpu_mode=args.cpu,
              after_construct_launcher_callbacks=[register_evaluate_fn],
              opts=args.opts)
