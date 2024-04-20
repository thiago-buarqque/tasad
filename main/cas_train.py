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
# lr      = 0.0001
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

        for epoch in range(args.epochs):            
            sum_loss = 0

            widgets = [
                      DynamicMessage('epoch'),
                      Bar(marker='=', left='[', right=']'),
                      ' ',
                      ETA(),
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

            if args.visualize:
                visualizer.plot_loss(l2_loss, n_iter, loss_name='cas_l2_loss')
                visualizer.plot_loss(ssim_loss, n_iter, loss_name='cas_ssim_loss')

                visualizer.visualize_image_batch(train_batch, n_iter, image_name='cas_sample_input')
                visualizer.visualize_image_batch(aug_train_batch, n_iter, image_name='cas_sample_augmented')
                visualizer.visualize_image_batch(anomaly_mask_batch, n_iter, image_name='cas_sample_gt')
                visualizer.visualize_image_batch(prediction, n_iter, image_name='cas_out_pred')

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

                except Exception as e:
                    print(e)

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