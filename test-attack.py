from attack import Attack
#from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR, CIFARModel
import os
import utils

args = {}
filelist = os.listdir('images')

utils.load_image(filelist.index('cifar_test_data_10.jpg'), 'images')

#print(filelist[0])
args['image_id'] = filelist.index('cifar_test_data_10.jpg')
args['targeted_attack'] = 0
args['folder'] = 'Images'
model = CIFARModel("models/cifar", None, True)
args['model'] = model

'''
From other code:
arg_init_const = args['init_const']
arg_kappa = args['kappa']
arg_q = args['q']
arg_mode = args['mode']
arg_save_iteration = args['save_iteration']
'''

args['init_const'] = .1 # different for mnist
args['kappa'] = 1e-10
args['q'] = 10
args['targeted_attack'] = False
args['save_iteration'] = False
args['grad_descent'] = "ZO-signSGD"

new = Attack(args)
new.generate_attack()
#new.get_orig_values()
