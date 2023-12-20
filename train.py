import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from collections import OrderedDict

def cal_val_pfm_idx(mtpfm_val_list):
    mse_sum = 0
    psnr_sum = 0
    ssim_sum = 0
    data_len = len(mtpfm_val_list)
    for mtpfm in mtpfm_val_list:
        mse_sum += mtpfm['MSE']
        psnr_sum += mtpfm['PSNR']
        ssim_sum += mtpfm['SSIM']
    mse = mse_sum / data_len
    psnr = psnr_sum / data_len
    ssim = ssim_sum / data_len
    
    mt_pfm = OrderedDict()
    mt_pfm['MSE'] = mse
    mt_pfm['PSNR'] = psnr
    mt_pfm['SSIM'] = ssim
    return mt_pfm
    
         
if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations


        
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                # mtpfm = visualizer.cal_current_pfm(epoch,model.get_current_visuals())
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                    #visualizer.plot_current_ssim(epoch, float(epoch_iter) / dataset_size, mtpfm)
                    

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

            # Add Validation
            val_opt = TestOptions().parse()
            val_opt.batch_size = 1
            
            data_val = create_dataset(val_opt)
            val_dataset_size = len(data_val)
            
            model_val = create_model(val_opt)      # create a model given opt.model and other options
            model_val.setup(val_opt)               # regular setup: load and print networks; create schedulers
            #model_val.eval() 
            mtpfm_val_list = []   
            mse_total = 0
            ssim_total = 0
            psnr_total = 0
            count = 0
            for i, data in enumerate(data_val):
                count += 1
                model_val.set_input(data)  # unpack data from data loader
                model_val.test()           # run inference
                val_visuals = model_val.get_current_visuals() 
                mtpfm_val_od = visualizer.cal_current_pfm(epoch,val_visuals)
                mse_total += mtpfm_val_od['MSE']
                psnr_total += mtpfm_val_od['PSNR']
                ssim_total += mtpfm_val_od['SSIM']
            
            mse_avg = mse_total / count
            ssim_avg = ssim_total / count
            psnr_avg = psnr_total / count
            
            mtpfm_val = OrderedDict()
            mtpfm_val['MSE'] = mse_avg
            mtpfm_val['SSIM'] = ssim_avg
            mtpfm_val['PSNR'] = psnr_avg
            
            visualizer.plot_current_ssim_val(epoch, float(epoch_iter) / dataset_size, mtpfm_val)
            t_comp = (time.time() - epoch_start_time) / opt.batch_size
            t_data = time.time() - epoch_start_time
            visualizer.print_current_val_mtx(epoch, t_comp, t_data, mtpfm_val)
            
            
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
