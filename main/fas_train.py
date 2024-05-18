import torch
from test_seg_model import test_seg_model
from seg_model import *
from torch import optim
import os
from utils.tensorboard_visualizer import TensorboardVisualizer
from torch.utils.data import DataLoader
from data_loaders.data_loader import MVTecTrainDataset
from utils.utilts_func         import *
from seg_model import *
# from loss.loss import SSIM
from loss.focal_loss import *
from loss.loss import SSIM
from tqdm import tqdm
import time
from progressbar import Bar, DynamicMessage, ProgressBar, ETA

def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #CUDA_LAUNCH_BLOCKING=1

    for class_name in args.class_name[0].replace("['", "").replace("']", "").split(","):
        args.train_gpu_id   =  ','.join([str(i) for i in args.train_gpu_id])
        cuda_map            =  f'cuda:{args.train_gpu_id}'
        cuda_id             =  torch.device(cuda_map) 
        base_model_name     = 'fas_seg_model_weights_mvtech_'

        visualizer      = TensorboardVisualizer(log_dir=os.path.join(args.log_path, base_model_name+class_name+ "/"))    
        ### cas model
        # cas_model       = Seg_Network(in_channels=3, out_channels=1)
        # cas_par         = Seg_Network.get_n_params(cas_model)/1000000 
        
        # print("Number of Parameters of CAS model: ", cas_par, " million")
        
        # cas_model.cuda(cuda_id)
        # cas_model.load_state_dict(torch.load(f'{args.checkpoint_cas_weights}{class_name}.pckl', map_location=f'{cuda_map}'))
        
        ### fas model
        fas_model       = Seg_Network(in_channels=3, out_channels=1)
        fas_par         = Seg_Network.get_n_params(fas_model)/1000000 
        
        print("Number of Parameters of FAS model:", fas_par, " million")
        
        fas_model.cuda(cuda_id)
        
        if args.checkpoint_fas_weights=='':
            fas_model.apply(weights_init)
        else:
            fas_model.load_state_dict(torch.load(f'{args.checkpoint_fas_weights}{class_name}.pckl', map_location=f'{cuda_map}'))
   

        optimizer = torch.optim.Adam([{"params": fas_model.parameters(), "lr": args.lr}])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)


        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM(args.train_gpu_id)
        #loss_focal = BinaryFocalLoss()

        ### dataset 
        dataset     = MVTecTrainDataset(f'{args.data_path}{class_name}/train/', args.anomaly_source_path, args.anom_choices, resize_shape=[256, 256],include_norm_imgs=1, datatype=f'{args.datatype}')
        dataloader  = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=12) # 16

        last_best_loss = 1e10
        last_best_epoch = 1

        for epoch in tqdm(range(args.epochs)):
            
            sum_loss = 0
            sum_ssim_loss = 0
            sum_l2_loss = 0

            widgets = [
                      DynamicMessage('epoch'),
                      Bar(marker='=', left='[', right=']'),
                ' ',  ETA(),
            ]
            with ProgressBar(widgets=widgets, max_value=len(dataloader)) as progress_bar:
                # time_batches = []
                # time_evals = []
                # time_backward = []
                # time_losses = []
                for i_batch, sample_batched in enumerate(dataloader):
                    # start_batch = time.time()
                    # input_batch         = sample_batched["image"].cuda(cuda_id)
                    aug_batch           = sample_batched["augmented_image"].cuda(cuda_id)
                    groudtruth_mask     = sample_batched["anomaly_mask"].cuda(cuda_id)

                    # cas_output          = cas_model(aug_batch)
                    # input_batch_fas     = torch.tensor(seg_module(aug_batch,cas_output,th_pix=1, th_val=0)).cuda(cuda_id)
                    # start_eval = time.time()
                    fas_output          = fas_model(aug_batch)
                    
                    # end_eval = time.time()
                    # time_evals.append(end_eval - start_eval)

                    # start_loss = time.time()
                    l2_loss             = loss_l2(fas_output,groudtruth_mask)
                    
                    sum_l2_loss += l2_loss
                    
                    ssim_loss           = loss_ssim(fas_output, groudtruth_mask)
                    
                    sum_ssim_loss += ssim_loss
                    
                    #segment_loss        = loss_focal(fas_output, groudtruth_mask) 
                    # loss                = l2_loss  #+ ssim_loss #+ segment_loss
                    loss = l2_loss + ssim_loss
                    
                    sum_loss += loss
                    # end_loss = time.time()
                    
                    # time_losses.append(end_loss - start_loss)

                    # start_backward = time.time()
                    optimizer.zero_grad()
                    
                    loss.backward()
                    
                    optimizer.step()
                    
                    # end_backward = time.time()
                    
                    # time_backward.append(end_backward - start_backward)
                    
                    # end_batch = time.time()
                    
                    # time_batches.append(end_batch - start_batch)
                    
                    progress_bar.update(
                        i_batch, 
                        epoch=f"({epoch+1}) Class: {class_name} | L2 loss: {(sum_l2_loss / (i_batch + 1)):.2f} | SSIM loss: {(sum_ssim_loss / (i_batch + 1)):.2f} | L2 and SSIM loss: {(sum_loss / (i_batch + 1)):.2f} ")
                    
                # time_batches = np.array(time_batches)
                # time_evals = np.array(time_evals)
                # time_backward = np.array(time_backward)
                # time_losses = np.array(time_losses)
                # print(f"Time avg for batch: {time_batches.mean()} | eval: {time_evals.mean()} | back: {time_backward.mean()} | losses: {time_losses.mean()}")
                
            # start = time.time()
            
            # torch.save(fas_model.state_dict(), os.path.join('./test_weights/', f"{base_model_name}{class_name}.pckl"))
            
            
            # end = time.time()
            # print(f"Time spent to save model: {(end - start) *  1000}")
            
            avg_loss = sum_loss / len(dataloader)
            if avg_loss < last_best_loss and abs(avg_loss - last_best_epoch) >= 1e-3:
                last_best_loss = avg_loss
                last_best_epoch = epoch + 1
                
                torch.save(fas_model.state_dict(), os.path.join(args.best_model_save_path, f"{base_model_name}{class_name}.pckl"))
                
            
            # print(f"Train loss avg: {avg_loss:.2f}", end=" ")
            
            visualizer.plot_loss(sum_l2_loss / len(dataloader), epoch, loss_name='fas_l2_loss')
            visualizer.plot_loss(sum_ssim_loss / len(dataloader), epoch, loss_name='ssim_loss')
            # visualizer.plot_loss(segment_loss, epoch, loss_name='focal_loss')
            
            # visualizer.visualize_image_batch(aug_batch, epoch, image_name='fas_cas_input')
            # visualizer.visualize_image_batch(cas_output, epoch, image_name='fas_cas_output')
            # visualizer.visualize_image_batch(input_batch, epoch, image_name='fas_input_batch')
            visualizer.visualize_image_batch(aug_batch, epoch, image_name='fas_input')
            visualizer.visualize_image_batch(groudtruth_mask, epoch, image_name='fas_mask_target')
            visualizer.visualize_image_batch(fas_output, epoch, image_name='fas_output')
            
            ap, ap_pixel, auroc, auroc_pixel = test_seg_model(
                fas_model,
                class_name,
                args.data_path,
                epoch + 1,
                args.val_gpu_id,
                None,
                visualizer
            )
            
            print(f"(test) AP: {ap:.2f} | AP pixel: {ap_pixel:.2f} | AUROC: {auroc:.2f} AUROC pixel: {auroc_pixel:.2f}")

            scheduler.step()
            # elif (epoch + 1) - last_best_epoch >= 15:
            #     break

            


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--train_gpu_id', action='store', type=str, default=0, required=False)
    parser.add_argument('--val_gpu_id', action='store', type=str, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store', type=str, required=True) #action='store_true')
    parser.add_argument('--checkpoint_cas_weights', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_fas_weights', action='store', type=str, required=True)
    parser.add_argument('--class_name', action='store', type=str,nargs='+', required=True)
    parser.add_argument('--anomaly_type', dest='anom_choices', required=False, help='0 -- for superpixel, 1 -- for perlin and [0,1] for both', nargs='+', type=int, default=[0,1])
    parser.add_argument('--cas_model_path', action='store', type=str, required=True)
    parser.add_argument('--datatype', action='store', type=str, required=True)
    parser.add_argument('--best_model_save_path', action='store', type=str, required=True)

    args = parser.parse_args()
    gpu_ids     =   args.train_gpu_id.split(',')
    gpu_id      = [int(i) for i in gpu_ids]
    args.gpu_id = gpu_id     

    with torch.cuda.device(args.gpu_id):
        train_on_device(args)