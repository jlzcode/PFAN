set -ex
python test.py --dataroot /opt/data/private/pytorch-CycleGAN-and-pix2pix-master_Mdf/datasets/c80_full_pix2pix  \
               --name c80_uformer_org_full \
               --model pix2pix \
               --netG uformer \
               --direction AtoB \
               --dataset_mode aligned \
               --norm batch \
               --phase test\
               --ndf 64 \
               --ngf 64  

               

