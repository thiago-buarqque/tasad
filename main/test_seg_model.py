import torch
from data_loaders.data_loader_test import MVTecTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt 
from utils.tensorboard_visualizer import TensorboardVisualizer
from utils.utilts_custom_class import *
from utils.utilts_func         import *
import cv2
from seg_model import *
from typing import Union
from datetime import datetime


def test_seg_model(
    cas_model: Seg_Network,
    class_name: str,
    data_path: str,
    epoch: int,
    gpu_id,
    fas_model: Union[Seg_Network, None] = None,
    visualizer: TensorboardVisualizer = None):
    
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []

    cuda_map        =  f'cuda:{gpu_id}'
    cuda_id         =  torch.device(cuda_map) 
    
    img_dim     = 256
    img_ind     = 0

    # cas_model   = Seg_Network(in_channels=3, out_channels=1)
    # cas_par     = Seg_Network.get_n_params(cas_model)/1000000 
    # print("Number of Parmeters of ReconstructiveSubNetwork", cas_par, "Million")

    # cas_model.cuda(cuda_id)
    # cas_model.load_state_dict(torch.load(checkpoint_path+args.model_name+".pckl", map_location=f'cuda:{args.gpu_id}'))
    # cas_model.cuda(cuda_id)
    cas_model.eval()

    ########### FAS model initialization ########## 
    if fas_model != None:
        # model_seg = Seg_Network(in_channels=3, out_channels=1)
        # model_seg.load_state_dict(torch.load(checkpoint_path+'fas'+args.model_name[3:]+'.pckl', map_location=f'cuda:{args.gpu_id}'))
        # model_seg.cuda(cuda_id)
        fas_model.eval()
    ########### FAS model initialization ended ########## 

    dataset     = MVTecTestDataset(data_path + class_name + "/test/", resize_shape=[img_dim, img_dim], datatype='png')
    dataloader  = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores          = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores       = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt            = []
    anomaly_score_prediction    = []

    for i_batch, sample_batched in enumerate(dataloader):
        cas_input               = sample_batched["image"].cuda(cuda_id)

        orig_image              =   plt.imread(dataloader.dataset.images[i_batch]) 
        orig_img_res            =   cv2.resize(orig_image, (img_dim, img_dim))

        is_normal       = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
        anomaly_score_gt.append(is_normal)
        true_mask       = sample_batched["mask"]
        
        true_mask_cv    = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
        cas_out         = cas_model(cas_input)
            
        out_mask_cv     = cas_out[0 ,0 ,: ,:].detach().cpu().numpy()
        output          = cas_out                                                                                                                                   
        
        ########### FAS model applied to output of CAS      ########## 
        if fas_model != None:
            fas_input                   =   torch.tensor(seg_module(cas_input,cas_out,th_pix=0.95, th_val=30)).cuda(cuda_id) # recommended value : gray_rec,th_pix=1, th_val=0
            
            fas_output                  =   fas_model(fas_input) #gray_batch 
            cas_fas_ouput               =   fas_output + cas_out 
        ########### FAS model applied to output of CAS ended ########## 

            out_mask_cv = cas_fas_ouput[0 ,0 ,: ,:].detach().cpu().numpy()
            output      = cas_fas_ouput                                                                                                                                   
        #plt.imsave(f'./results_cutom/actual_{class_name}{i}.png',np.hstack((out_mask_cv,true_mask_cv[:,:,0])))
        try:
            fas_input               =    fas_input.detach().cpu().numpy()[0, :, :, :].transpose((1, 2, 0))   
            fas_input               =    cv2.cvtColor(fas_input, cv2.COLOR_BGR2RGB)
            query_image_cv          =    cv2.cvtColor(orig_img_res, cv2.COLOR_BGR2RGB)
            out_mask_fas            =    abs(fas_output.detach().cpu().numpy()[0, :, :, :].transpose((1, 2, 0))[:,:,0])/torch.max(fas_output).item()
            out_mask_cas            =    abs(cas_out.detach().cpu().numpy()[0, :, :, :].transpose((1, 2, 0))[:,:,0])/torch.max(cas_out).item()
            cas_fas_ouput_n         =    abs(out_mask_fas + out_mask_cas)
            cas_fas_ouput_n         =    cas_fas_ouput_n/np.max(cas_fas_ouput_n)
            ''' cas_fas_ouput_colr      =    float_int8_color(cas_fas_ouput_n)
            input_img_with_mask     =    (query_image_cv*255) + cv2.cvtColor(cas_fas_ouput_colr, cv2.cv2.COLOR_BGR2HSV)/255  '''
            
            all_imgs             =   [query_image_cv,  true_mask_cv, out_mask_cas, fas_input, out_mask_fas, cas_fas_ouput_n] 
            # img_names            =  '' #'[ "Query Image","Ground Truth Mask","Predicted Mask"]'
            pred_result_save(all_imgs, save_res= True) #show_results= True)

            #cv2.imwrite(f'./results_cutom/actual_{class_name}{i}.png',np.hstack((np.array((out_mask_cv/np/max(out_mask_cv))*255, dtype=np.uint8),np.array(true_mask_cv[:,:,0]*255, dtype=np.uint8))))
        except Exception as e:
            #print("Error in saving func", e)
            pass
        img_ind+=1
        out_mask_averaged   = torch.nn.functional.avg_pool2d(output[: ,: ,: ,:] , 21, stride=1,
                                                            padding=21 // 2).cpu().detach().numpy() # chnage  # chnage 
                                                            # orig: out_mask_sm[: ,1: ,: ,:] final_img1
        image_score = np.max(out_mask_averaged) # -- global max pooling 

        anomaly_score_prediction.append(image_score)

        flat_true_mask              = true_mask_cv.flatten()
        flat_out_mask               = out_mask_cv.flatten()
        
            
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    
    obj_ap_pixel_list.append(ap_pixel)
    obj_auroc_pixel_list.append(auroc_pixel)
    obj_auroc_image_list.append(auroc)
    obj_ap_image_list.append(ap)
        
    if visualizer != None:
        # prefix = "fas" if fas_model != None else "cas"
        visualizer.plot_performance(score_val=ap_pixel, n_iter=epoch, metric_name=f"test_AP_pixel")
        visualizer.plot_performance(score_val=auroc_pixel, n_iter=epoch, metric_name=f"test_AUROC_pixel")
        visualizer.plot_performance(score_val=auroc, n_iter=epoch, metric_name=f"test_AUROC")
        visualizer.plot_performance(score_val=ap, n_iter=epoch, metric_name=f"test_AP")
    
    print(f"{datetime.now()} Test for epoch {epoch}: Class {class_name} Pixel AP {ap_pixel:.2f} Pixel AUC {auroc_pixel:.2f} Image AUC {auroc:.2f} Image AP {ap:.2f}")
    
    return ap, ap_pixel, auroc, auroc_pixel


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--both_model', action='store', type=int, required=True)
    parser.add_argument('--obj_list_all', action='store', help='class names as a list', type=str,required=True)
    args = parser.parse_args()
    
    try:
        obj_list = [item for item in args.obj_list_all.split(',')]
    except:
        obj_list = [args.obj_list_all]

    class_name = obj_list[0]

    with torch.cuda.device(args.gpu_id):
        # cas_model   = Seg_Network(in_channels=3, out_channels=1)
        # cas_model.load_state_dict(torch.load(args.checkpoint_path + args.model_name+class_name+".pckl", map_location=f'cuda:{args.gpu_id}'))
        # cas_model.cuda(args.gpu_id)
        
        fas_model = Seg_Network(in_channels=3, out_channels=1)
        fas_model.load_state_dict(torch.load(args.checkpoint_path+'fas'+args.model_name[3:]+class_name+'.pckl', map_location=f'cuda:{args.gpu_id}'))
        fas_model.cuda(args.gpu_id)

        test_seg_model(
            fas_model,
            class_name,
            args.data_path, 
            -1,
            args.gpu_id,
            None
        )
