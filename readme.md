# TASAD workflow 
![fasad_model](https://github.com/RizwanAliQau/tasad/assets/29249233/7090cb30-663a-4e6a-ae27-a35b2f65793e)

# Novel SAIM (Superpixel based anomaly insertion method) -----> generates three different size of anomalies for model generalization
### The pseudo-anomaly insertion by the proposed SAIM
![saim-1](https://github.com/RizwanAliQau/tasad/assets/29249233/88ffd8aa-ed87-4da0-9e0b-cf55e4f80b4c)

### Example of pseudo-anomaly insertion by SAIM: generates different sizes of anomalies i.e., small, medium and large by controlling the number of segments of anomaly source image.
![Size_of_anom-1](https://github.com/RizwanAliQau/tasad/assets/29249233/371e2bf9-9c8a-44d7-98e3-8e8823fd4b71)


# create directories 
    
    ├── best_weights_model_1
    ├── best_weights_model_2
    ├── checkpoints
    ├── data
    ├── logs
    ├── test_weights
    └── weights
# create conda environment
    conda create -n ENVNAME --file requirement.txt
    pip install -r requirement_pip.txt

# for testing 
    - download the weights
        - https://drive.google.com/drive/folders/10Z0MNGY9codk0F-h59roTUr4Xeay2IPO?usp=share_link

# tasad testing 
    python ./main/test_seg_model.py --gpu_id 0 --model_name cas_seg_model_weights_mvtech_ --data_path ./data/images/ --checkpoint_path ./test_weights/ --both_model 1 --obj_list_all zipper

# for training 
    - download mvtec dataset
        - https://www.mvtec.com/company/research/datasets/mvtec-ad
    - download the texture dataset put inside anomaly source image 
        - https://www.robots.ox.ac.uk/~vgg/data/dtd/
# cas training 
    python ./main/cas_train.py --gpu_id 0 --gpu_id_validation 0 --obj_id -1 --lr 0.0001 --bs 4 --epochs 2 --data_path ./data/images/ --anomaly_source_path ./data/anomaly/images/ --checkpoint_path ./test_weights/ --log_path ./logs/ --checkpoint_cas_model "" --visualize True --class_name hazelnut --best_model_save_path ./best_weights_model_1/

# fas training 
    python ./main/fas_train.py --train_gpu_id 0 --val_gpu_id 0 --obj_id -1 --lr 0.0001 --bs 2 --epochs 200 --data_path ./data/images/ --anomaly_source_path ./data/anomaly/images/ --cas_model_path ./test_weights/ --checkpoint_path ./checkpoints/ --log_path ./logs/ --checkpoint_cas_weights ./test_weights/cas_seg_model_weights_mvtech_ --checkpoint_fas_weights "" --visualize True --datatype png --best_model_save_path ./best_weights_model_2/ --class_name bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper

##### ----- #### 
Thanks DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection ---> https://github.com/VitjanZ/DRAEM for providing their code and model weights  


python3 ./main/data_processor.py --data_path ./data/anomaly/images/ --anomaly_source_path ./data/anomaly/images/ --class_name bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper