### training ###

# for single GPU
python train.py -opt=options/train/ReCoFuse_train.yml

# for multiple GPUs
torchrun --nproc_per_node=2 --master_port=6115 train.py -opt=options/train/ReCoFuse_train.yml --launcher pytorch

### testing ###
python test.py -opt=options/test/ReCoFuse_test.yml

