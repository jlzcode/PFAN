set -ex
python test.py --dataroot /datasets/c80  \
               --name c80 \
               --model pix2pix \
               --netG pfan \
               --direction AtoB \
               --dataset_mode aligned \
               --norm batch \
               --phase test\
               --ndf 64 \
               --ngf 64  

               

