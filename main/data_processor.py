import numpy as np
from pathlib import Path
#from py import process
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from utils.utilts_func import *
from utils.perlin import rand_perlin_2d_np
from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from progressbar import Bar, DynamicMessage, ProgressBar, Percentage, GranularBar, \
    Timer, ETA, Counter

class DataProcessor(Dataset):

    def __init__(
        self,
        root_dir,
        class_name,
        anomaly_source_path,anom_choices=[0,1],
        resize_shape=None,
        include_norm_imgs=0,
        datatype='png'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.class_name = class_name

        self.image_paths = sorted(glob.glob(root_dir+ f"*.{datatype}")) #"*.png")) ### change /*/*.jpg
        
        if len(self.image_paths) == 0: 
            # print(f"Looking on {root_dir}/*/*.{datatype}")
            self.image_paths = sorted(glob.glob(root_dir+ f"/*/*.{datatype}"))
            
        # print(f"Found {len(self.image_paths)}")
        
        # print(f"Image paths: {self.image_paths}")

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+'/*/*.png'))
        
        if len(self.anomaly_source_paths)==0:
            self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+'/*/*.jpg')) # "/*/*.png"))
            
        if len(self.anomaly_source_paths)==0:
            self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+'*.jpg'))

        if include_norm_imgs!=0:
            self.anomaly_source_paths = sorted(add_norm_imgs_to_anom_imgs(self.anomaly_source_paths, self.image_paths))
        
        self.anomaly_type = anom_choices        
       
        self.augmenters = [
                iaa.GammaContrast((0.5,2.0),per_channel=True), ## per_channel to handle tghe bightness effect
                iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)), ### brightness mult+ add
                iaa.pillike.EnhanceSharpness(), ## randomly increase or decrease sharpness
                iaa.pillike.Autocontrast(),
            ]

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )

        return aug

    def augment_image(self, image, anomaly_source_path, anomaly_type=[0,1]):
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5 or anomaly_type==2:
            # image = image #.astype(np.float32)/255
            return image, np.expand_dims(np.zeros_like(image[:,:,0], dtype=np.float32),axis=2), np.array([0.0],dtype=np.float32)
        
        aug = self.randAugmenter()
        
        perlin_scale = 6
        min_perlin_scale = 0
        #anomaly_type = 0 
        
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        #anomaly_img_augmented = anomaly_source_img
        
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        
        """
        #perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        #perlin_noise = self.rot(image=perlin_noise) ### riz change 
        # threshold = 0.5
         perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)
        """
        
        ### new anomaly 
        choice_for_anomly   = np.random.choice(self.anomaly_type,1)[0] # (2,2)
        
        """ Seg Choices
            [150,200,250]
            [5,10,20,30,40,50]
            [10,20,30,40,50,60,70]
            [10,20,30,40,50,60,70,150,200,250]
        """
        seg_choice          = [30, 150, 250]
        
        choice_for_seg_anom = np.random.choice(seg_choice,len(seg_choice))[0]
        
        if choice_for_anomly==0: ## TODO: correct it 
            choice_for_rotate       = np.random.choice(3,3)[0]
            
            if choice_for_rotate==0:
                anomaly_img_augmented    = cv2.rotate(anomaly_img_augmented, cv2.ROTATE_90_CLOCKWISE)
            elif choice_for_rotate==1:
                anomaly_img_augmented    = cv2.rotate(anomaly_img_augmented, cv2.ROTATE_180)
            
            #t1                      =       timeit.default_timer()
            segments_slic           =       slic(anomaly_img_augmented.reshape((256,256,3)), n_segments=choice_for_seg_anom, compactness=8, sigma=1,start_label=1)
            no_of_seg               =       np.unique(segments_slic)
            uniq_v                  =       np.random.choice(no_of_seg, int(len(no_of_seg)*0.1)) # uniq_v = np.random.choice(no_of_seg)   
            
            seg_img                 =       np.isin(segments_slic,uniq_v)
            msk                     =       np.expand_dims(seg_img, axis=2)
            # augmented_image         =       anomaly_img_augmented.astype(np.float32) /255              
            augmented_image         =       anomaly_img_augmented.astype(np.float32)         
            augmented_image         =       msk * augmented_image + (1-msk)*image
            #t2                      =       timeit.default_timer()

            #print("processing time for anomaly insertion ", t2-t1)
            #perlin_thr              =       np.expand_dims(perlin_thr, axis=2)
            #augmented_image         =       image *(1-perlin_thr)+perlin_thr    
            
        elif choice_for_anomly==1:  #1
            perlin_noise        = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            perlin_thr          = np.where(perlin_noise >  np.random.rand(1)[0], np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            
            #perlin_thr = np.where((anomaly_img_augmented.astype(np.float32)[:,:,0]/255)>0.5, np.ones_like(perlin_noise), np.zeros_like(perlin_noise)) 
            #perlin_thr = np.where(perlin_noise >  np.random.rand(1)[0], np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr          = np.expand_dims(perlin_thr, axis=2)
            msk                 = np.copy(perlin_thr)
            # img_thr             = (anomaly_img_augmented.astype(np.float32) * perlin_thr) / 255.0
            img_thr             = (anomaly_img_augmented.astype(np.float32) * perlin_thr)

            beta                = torch.rand(1).numpy()[0] * 0.8

            augmented_image     = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
                perlin_thr)
        ''' else:
            image           = aug(image=np.array(image*255, dtype=np.uint8))
            anomaly_type    = 2  '''
            
        ### new anomaly end

        ### added for new anomaly
        #augmented_image = augmented_image.astype(np.float32)
        try:
            augmented_image = augmented_image.astype(np.float32)
        except: pass
            
        msk = (msk).astype(np.float32)
        
        if len(msk.shape)==2:
            # dim    =     msk.shape[0] 
            msk    =     np.expand_dims(msk, axis=2)

        ### added for new anomaly [Segmentation]
        dec_brig   = 1
        
        if self.anomaly_type==[0]:
            patch_aug   =  msk * augmented_image *dec_brig
            patch_img   =  msk * image
            ssim_value  =  ssim(patch_aug, patch_img, multichannel=True) 
            if ssim_value>0.85 and self.anomaly_type==[0,1]:
                #msk             = msk*0.0
                #augmented_image = image 
                perlin_noise                = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                anomaly_img_augmented       = anomaly_source_img
                augmented_image,perlin_thr  = per_anomaly(perlin_noise, anomaly_img_augmented, image)
                augmented_image             = augmented_image.astype(np.float32)
                msk                         = (perlin_thr).astype(np.float32)
                
            #augmented_image         = msk * augmented_image*dec_brig + (1-msk)*image
        
        has_anomaly = 1.0

        if np.sum(msk) == 0:
            has_anomaly=0.0

        return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        ### riz change 
        '''
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)
        '''
        
        # image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32)
        
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        
        image = np.transpose(image, (2, 0, 1))
        
        return image, image, None, False
        
        ### added for new anomaly 
        if len(anomaly_mask.shape)==2:
            dim             =   anomaly_mask.shape[0]
            anomaly_mask    =   np.expand_dims(anomaly_mask, axis=2)
        
        ### added for new anomaly
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        
        return image, augmented_image, anomaly_mask, has_anomaly

    def __save_image(self, image, path, path_prefix="./processed_data"):
        if path.startswith("./data"):
            path = path[6:]
        
        file_path = path_prefix + path
        
        dir_path: str = os.path.dirname(file_path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path) 
            
        if image.shape[0] == 4:
            image = image[:, :3, :, :]
            
        image_transposed = image.transpose(1, 2, 0)
        
        if image_transposed.max() > 1.0:
            image_transposed = image_transposed.astype(np.float32) / 255.0

        # fig, ax = plt.subplots()
        # fig.patch.set_facecolor('black')
        # ax.set_facecolor('black')

        if image_transposed.shape[2] == 1:
            image_transposed = np.repeat(image.squeeze(), 3).reshape((image_transposed.shape[0], image_transposed.shape[1], 3))

        plt.imsave(file_path, image_transposed)

        # ax.imshow(image_transposed)

        # Turn off the axes
        # ax.axis('off')

        # Save the figure
        # plt.savefig(file_path, bbox_inches='tight', pad_inches=0, facecolor='black', transparent=False)

        # Close the plot to free memory
        # plt.close(fig)

    def __len__(self):
        return len(self.image_paths)
    
    def process_image(self, idx):
        # idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        
        path = self.image_paths[idx]
        anomaly_path = self.anomaly_source_paths[anomaly_idx]
        
        image, augmented_image, anomaly_mask, has_anomaly = \
            self.transform_image(path, anomaly_path)

        # self.__save_image(image, path, path_prefix="./processed_data/original")

        # self.__save_image(augmented_image, path, path_prefix="./processed_data/augmented")
        
        # path = "/" + path.split("/")[-1]
        # print(path)
        path = path.replace("./data", "")
        path = path.replace("./anomaly", "")
        path = path.replace(".png", ".jpg")
        
        if has_anomaly == 1.0:
            self.__save_image(augmented_image, path, path_prefix=f"./mvtec-256/{self.class_name}/test/with_anomaly")
            self.__save_image(anomaly_mask, path, path_prefix=f"./mvtec-256/{self.class_name}/ground_truth/with_anomaly")
        else:
            self.__save_image(augmented_image, path, path_prefix=f"./anomaly-256")

        return {
            'image': image, 
            'anomaly_mask': anomaly_mask,
            'augmented_image': augmented_image, 
            'has_anomaly': has_anomaly, 
            'idx': idx
        }


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, default='', required=True)
    parser.add_argument('--class_name', action='store', type=str,nargs='+', required=True)

    args = parser.parse_args()

    def process_image(data_processor, index):
        data_processor.process_image(index)

    widgets = [
                DynamicMessage('epoch'),
                Bar(marker='=', left='[', right=']'),
        ' ',  ETA(),
    ]

    for class_name in args.class_name[0].split(","):
        # print(f"Root: {args.data_path + class_name} anomaly: {args.anomaly_source_path}")
        dataset = \
            DataProcessor(
                root_dir=args.data_path + class_name + "/",
                anomaly_source_path=args.anomaly_source_path, 
                class_name=class_name,
                resize_shape=[256, 256],
                datatype="jpg"
            )
            
        with ProgressBar(widgets=widgets, max_value=dataset.__len__()) as progress_bar:
            for i in range(dataset.__len__()):
                dataset.process_image(i)

                progress_bar.update(
                        i,
                        epoch=f"({i}/{dataset.__len__()}) Class {class_name} ")