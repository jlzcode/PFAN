set -ex
python train.py --dataroot /opt/data/private/EnlightenGAN-master_Add_Uformer_20230405/c80_298  \
    --name comparative/c80_cycle_resnet\
    --model cycle_gan \
    --no_dropout
