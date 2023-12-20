set -ex
python train.py --dataroot  /opt/data/private/pytorch-CycleGAN-and-pix2pix-master_Mdf/datasets/c80_full_pix2pix  \
    --name c80_uformer_org_full_debug_1 \
    --model pix2pix \
    --netG pfan \
    --netD basic \
    --direction AtoB \
    --dataset_mode aligned \
    --norm batch \
    --checkpoints_dir /opt/data/private/pytorch-CycleGAN-and-pix2pix-master_Mdf3/checkpoints  \
    --token_projection 'conv'  \
    --embed_dim 64 \
    --ndf 64 \
    --ngf 64   \

    
        # --lambda_L1 100 \
        # --pool_size 0 \
    # --use_wandb \
    # --wandb_project_name c80_ConvNext_2b_2