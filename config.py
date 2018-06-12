config = {
      'model_dir': 'exps/<time>', # save model at this dir, in main.py, would replace '<time>' by current time.
      'phase': 'train', # [train, finetune, test], use to determine learning rate and whether to create tensorboard writer or not.
      'mode': 'train+finetune', # train, finetune, test, train+finetune
      'gpu': [0,1], # use [] for cpu
      'restore_iter': -1, 
      'lambda_pixel_loss_hole': 6, 
      'lambda_pixel_loss_valid': 1, 
      'lambda_perceptual_loss_out': 0.2, 
      'lambda_perceptual_loss_comp': 0.2,
      'lambda_style_loss_out': 2000, 
      'lambda_gan': 0.2,
      'lambda_style_loss_comp': 2000, 
      'lambda_tv_loss': 0.1, 
      'lambda_local_gan': 1, # weight of local GAN: local GAN would find image patch(size=patch_size) that contains masked pixels as discriminator's input
      'finetune_iter': 100000, 
      'sample_freq': 500, # save training samples every #sample_freq iterations
      'save_freq': 5000, # save model every #save_freq iterations
      'vgg_model': 'vgg_16.pth', 
      'train_lr': 2e-4, 
      'finetune_lr': 5e-5,
      'gan_loss': 'mse', # mse, bce
      'use_gan': True, 
      'use_tv_loss': True, 
      'use_vgg': True, # if True, would compute vgg loss: style loss and perceptual loss
      'g_input': 'masked_X+mask', # masked_X, masked_X+mask
      'pix2pix_style': False, # if True, use [masked_X, X_gt] as real sample and [masked_X, inpainted_X] as fake samples for discriminator
      'decoder_partial_conv': False, # whether use partial conv in decoder
      'init_type': 'kaiming', # xavier, normal, kaiming
      'patchgan': False,  # if true, global gan would use patchgan 
      'use_local_d': True, # always use gan, when current_resol>=config['patch_size'] and use_local_d=True, will used local gan
      'patch_size': 128, # patch size of image patches for local GAN
      'start_gan_at': 0, # start using GAN at which iteration?
      'learning_residual': True, # if True, the network would generate residual image, and use clip(masked_X + G(masked_X), 0, 1) as inpainted image
      'block': 'basic', # basic(use upsample+conv), residual
      'progressive_growing': True, # if True, use progressive growing training scheme, see https://arxiv.org/abs/1710.10196. Warning: the implementation might be problematic.
      'from_resol': 64, # if progressive_growing=True, would start at this resolution
      'to_resol': 512, # target resolution
      'n_real_per_phase': 800000, # number of real samples seen at per phase, this is used to determine how many iterations for each resolution & phase(stabilize, fade in)
      'bs_map': {4:32, 8:32, 16:16, 32:16, 64:16, 128:8, 256:4, 512:2}, # batch size of each resolution
      # 'bs_map': {4:32, 8:32, 16:16, 32:16, 64:16, 128:8, 256:6, 512:4},
}