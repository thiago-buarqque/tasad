import torch
from test_seg_model import test_seg_model
from seg_model import *
from torch import optim
from loss.loss import SSIM
from utils import *
import os
from utils.tensorboard_visualizer import TensorboardVisualizer
from torch.utils.data import DataLoader
from data_loaders.data_loader import MVTecTrainDataset
from utils.utilts_custom_class import *
from utils.utilts_func         import *
import subprocess
from loss.focal_loss import *
from progressbar import Bar, DynamicMessage, ProgressBar, Percentage, GranularBar, \
    Timer, ETA, Counter

### gloabal variables ----- arg
lr      = 0.0001
#pochs  = 800
###


def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    base_model_name = 'cas_seg_model_weights_mvtech_'
    
    for class_name in args.class_name:
        cuda_map        =  f'cuda:{args.gpu_id}'
        cuda_id         =  torch.device(cuda_map) 
        wght_file_name  =  base_model_name+class_name

        visualizer      = TensorboardVisualizer(log_dir=os.path.join(args.log_path, wght_file_name+"/"))    
        cas_model       = Seg_Network(in_channels=3, out_channels=1)
        cas_params      = Seg_Network.get_n_params(cas_model)/1000000 
        print("Number of parmeters of CAS model: ", cas_params, " million")

        cas_model.cuda(cuda_id)

        if args.checkpoint_cas_model=='':
            cas_model.apply(weights_init)
        else:
            cas_model.load_state_dict(torch.load(args.checkpoint_cas_model, map_location=cuda_map))  ##'cuda:0'))

        optimizer       = torch.optim.Adam([{"params": cas_model.parameters(), "lr": args.lr}])
        scheduler       = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_l2         = torch.nn.modules.loss.MSELoss()
        loss_ssim       = SSIM(args.gpu_id)
        
        dataset         = MVTecTrainDataset(args.data_path+class_name+'/train' , args.anomaly_source_path, resize_shape=[256, 256])
        dataloader      = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=12) # 16

        n_iter          = 0
        # prev_pixel_ap   = 0
        # prev_pixel_auc  = 0
        # prev_image_auc  = 0

        for epoch in range(args.epochs):            
            sum_loss = 0
            widgets = [
                      DynamicMessage('epoch'),
                      Bar(marker='=', left='[', right=']'),
                ' ',  ETA(),
            ]
            with ProgressBar(widgets=widgets, max_value=len(dataloader)) as progress_bar:
                for i_batch, batch in enumerate(dataloader):

                    train_batch             = batch["image"]
                    aug_train_batch         = batch["augmented_image"].cuda(cuda_id)
                    anomaly_mask_batch      = batch["anomaly_mask"].cuda(cuda_id)

                    prediction             = cas_model(aug_train_batch)

                    l2_loss                 = loss_l2(prediction,anomaly_mask_batch)
                    ssim_loss               = loss_ssim(prediction, anomaly_mask_batch)

                    loss                    = l2_loss + ssim_loss
                    
                    sum_loss += loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    progress_bar.update(
                        i_batch, 
                        epoch=f"({epoch}) Train loss: {(sum_loss / (i_batch + 1)):.2f}: Class {class_name} ")

            # It's so weird that the model is evaluated after n batches.
            # It should be after every epoch and with a validation set.
            if args.visualize:
                visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                # visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')               

                visualizer.visualize_image_batch(train_batch, n_iter, image_name='batch_input')
                visualizer.visualize_image_batch(aug_train_batch, n_iter, image_name='batch_augmented')
                visualizer.visualize_image_batch(anomaly_mask_batch, n_iter, image_name='ground_truth')
                visualizer.visualize_image_batch(prediction, n_iter, image_name='out_pred')

                torch.save(cas_model.state_dict(), os.path.join(args.checkpoint_path, f"{wght_file_name}.pckl"))

                try:
                    test_seg_model(
                        cas_model,
                        class_name,
                        args.data_path,
                        epoch,
                        args.gpu_id,
                        None,
                        visualizer
                    )
                    # results_val             = subprocess.check_output(f'python3 ./main/test_seg_model.py --gpu_id {args.gpu_id_validation} --model_name  {wght_file_name} --data_path {args.data_path} --checkpoint_path {args.checkpoint_path} --both_model 0 --obj_list_all {class_name}', shell=True)
                    # results_val             = decode_output(results_val)
                    # curr_pixel_ap,indx      = find_values(results_val, 'AP')
                    # curr_pixel_auc,_        = find_values(results_val, 'AUC')
                    # curr_image_auc,_        = find_values(results_val[indx:], 'AUC')

                    # if ((curr_pixel_auc+curr_pixel_ap+curr_image_auc)/3)>=((prev_pixel_ap+prev_pixel_auc+prev_image_auc)/3):
                    #     torch.save(cas_model.state_dict(), os.path.join(f"{args.best_model_save_path}", wght_file_name+".pckl"))
                    #     prev_pixel_ap           = curr_pixel_ap
                    #     prev_pixel_auc          = curr_pixel_auc
                    #     prev_image_auc          = curr_image_auc
                        
                    #     print("Saved pix AP value               :  ", prev_pixel_ap)
                    #     print("Saved pix AUC value              :  ", prev_pixel_auc)
                    #     print("Saved img UC value               :  ", prev_image_auc)
                    
                    # print(f"Test for epoch {epoch}: Class {results_val[7]} Pixel AP {curr_pixel_ap:.2f} Pixel AUC {curr_pixel_auc:.2f} Image AUC {curr_image_auc:.2f}")

                except Exception as e:
                    print(e)
                    
                    # torch.save(cas_model.state_dict(), os.path.join(args.checkpoint_path, model_path))

            n_iter +=1

            scheduler.step()



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--gpu_id_validation', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, default='', required=False)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store', type=str, required=True) #action='store_true')
    parser.add_argument('--checkpoint_cas_model', action='store', type=str, required=True)
    parser.add_argument('--class_name', action='store', type=str,nargs='+', required=True)
    parser.add_argument('--best_model_save_path', action='store', type=str, required=True)

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        train_on_device(args)