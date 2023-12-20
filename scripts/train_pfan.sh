set -ex
python train.py --dataroot  /datasets/c80  \
    --name c80    \
    --model pix2pix \
    --netG pfan \
    --netD basic \
    --direction AtoB \
    --dataset_mode aligned \
    --norm batch \
    --checkpoints_dir /checkpoints  \
    --token_projection 'conv'  \
    --embed_dim 64 \
    --ndf 64 \
    --ngf 64   \

    
        # --lambda_L1 100 \
        # --pool_size 0 \
    # --use_wandb \
    # --wandb_project_name c80_ConvNext_2b_2