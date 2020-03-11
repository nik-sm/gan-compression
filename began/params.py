batch_size = 32    # --> Size of training batch
size = 128  # --> Size of images for training
num_ups = 4 # --> Number of upsampling layers
# test_folder = "data"  # --> Location of image folder
# train_epochs = 40  # --> Number of training epochs
log_every = 100  # --> Number of iterations after which to log

latent_dim = 64  # --> Dimensionality of latent codes
ngpu = 1  # --> Number of GPUs
num_filters = 128  # --> Number of filters to use in each layer

lr = 0.0001  # --> Learning rate for Adam optimizers
lr_update_step = 12000  # --> Half the learning rate after this many steps
# lr_min = 0.00002 # --> Minimum learning rate

gamma = 0.5  # --> Image diversity hyperparameter: [0,1]
prop_gain = 0.001  # --> Proportional gain for k

carry = False # --> Use disappearing residuals
