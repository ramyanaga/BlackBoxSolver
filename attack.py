import os
import sys
import tensorflow as tf
import numpy as np
import random
import time
from keras.layers import Lambda
from tqdm import tqdm
from PIL import Image
#from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR, CIFARModel

import utils as util

class Attack:

    def __init__(self, args): 
        """
        Args:
            image: integer representing image id in folder 
                   (assumes user has folder w >= 1 image, images being attacked)
            model: a :class `Model` instance
                   The model that should be fooled by the adversarial image
            grad_estimator: String
                    String that specifies object used to calculate result 
                    of gradient using specified estimation method
            grad_descent: String
                    String that specified object used to calculate one iteration
                    of gradient descent using specified gradient descent method

        """
        self.image_id = args['image_id']
        self.image_folder = args['folder'] # is there a way to default this to None?
        self.model = args['model']
        self.orig_img, self.orig_prob, self.orig_class = self.get_orig_values()
        
        #self.grad_estimator = args['grad_estimator']
        #self.grad_descent = args['grad_descent']
        
        self.d = self.orig_img.size
        self.mu = 0.01
        self.q = args['q'] # re-read what q is used for
        #self.I = args['max_iter'] # check if actually supposed to user specified?
        self.I = 800 # max iters
        self.kappa = args['kappa'] # re-read what kappa is used for
        self.const = args['init_const']
        # TODO: figure this out self.const = args['const']
         
        self.targeted_attack = args['targeted_attack'] # should be boolean?
        print(self.targeted_attack)
        self.target_label = None # shouldn't be used for non-targeted attacks
        
        self.orig_img, self.orig_prob, self.orig_class = self.get_orig_values()
        #self.result = result_obj TODO: make result class, initialize this to result object

    def get_orig_values(self):
        '''
        1. Load image from folder of images
        2. Caclulate original probability, class, string from model
        ''' 
        print("in get_orig_values")
        orig_image = util.load_image(self.image_id, self.image_folder)
        orig_prob, orig_class, orig_prob_str = util.model_prediction(self.model,
                                                    np.expand_dims(orig_image, axis=0))


        #if self.targeted_attack:
        #    self.target_label = np.remainder(orig_class + 1, 10)
        #else:
        #    self.target_label = orig_class
        
        # TODO: figure out whether I need to include original image, original target stuff

        #print("orig_prob: ", orig_prob)
        #print("orig_class: ", orig_class)
        #print("orig_prob_str: ", orig_prob_str)
       
        print("before return of get_orig_values")
        return orig_image, orig_prob, orig_class


    #def function_evaluation(x, kappa, target_label, const, model, orig_img, arg_targeted_attack):
    def function_evaluation(self, x):
        # x is img_vec format in real value: w
        img_vec = 0.5 * np.tanh(x)
        img = np.resize(img_vec, self.orig_img.shape)
        #orig_prob, orig_class, orig_prob_str = util.model_prediction(model, img)
        #orig_prob, orig_class, orig_prob_str = self.orig_prob, self.orig_class, self.orig_prob_str
        tmp = self.orig_prob.copy()
        print("tmp copy: ", tmp)
        tmp[0, self.target_label] = 0
        #tmp[0, target_label] = 0
        #if arg_targeted_attack:  # targeted attack
        #    Loss1 = self.const * np.max([np.log(np.amax(tmp) + 1e-10) - np.log(orig_prob[0, target_label] + 1e-10), -kappa])
        #else:  # untargeted attack
        #    Loss1 = self.const * np.max([np.log(orig_prob[0, target_label] + 1e-10) - np.log(np.amax(tmp) + 1e-10), -kappa])
        if self.targeted_attack:
            Loss1 = self.const * np.max([np.log(np.amax(tmp) + 1e-10) - np.log(orig_prob[0, target_label] + 1e-10), 
                                            -self.kappa])
        else:
            Loss1 = self.const * np.max([np.log(self.orig_prob[0, self.target_label] + 1e-10) 
                                            - np.log(np.amax(tmp) + 1e-10), -self.kappa])

        Loss2 = np.linalg.norm(img - orig_img) ** 2
        return Loss1 + Loss2

    '''
    # Elastic-net norm computation: L2 norm + beta * L1 norm
    def distortion(a, b):
        return np.linalg.norm(a - b)
    '''

    '''
    # random directional gradient estimation - averaged over q random directions
    def gradient_estimation(mu,q,x,d,kappa,target_label,const,model,orig_img,arg_mode,arg_targeted_attack):
        # x is img_vec format in real value: w
        m, sigma = 0, 100 # mean and standard deviation
        f_0=function_evaluation(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)
        grad_est=0
        for i in range(q):
            u = np.random.normal(m, sigma, (1,d))
            u_norm = np.linalg.norm(u)
            u = u/u_norm
            f_tmp=function_evaluation(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
            # gradient estimate
            if arg_mode == "ZO-M-signSGD":
                grad_est=grad_est+ np.sign(u*(f_tmp-f_0))
            else:
                grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/mu
        return grad_est
        #grad_est=grad_est.reshape(q,d)
        #return d*grad_est.sum(axis=0)/q
    '''
    
    def generate_attack(self):
        '''
        Steps:
            1. load specified dataset + associated model
               calculate probability, class, probability string(?) of original image prediction
               stuff to make sure that the image_id is actually the image_id in the dataset
               make sure true label is the same as the original image class

            2. d = number of optimization vars
               mu = smoothing parameter
               q = number of random directions used to construct gradient estimate
               I = number of iterations
               kappa = parameter that controls gap bw confidence of correct class and second highest class
               const = idk what the constant is for???
               learning rates
               orig_image_vec
               w_img_vec
               initialize stuff that'll get set in result object

            3. actual iteration stuff (combine w/ 2)
                - split targeted vs. non-targeted attack into function bc v similar
       
        '''
        print("in generate_attack")
        ## flatten image to vec
        orig_img_vec = np.resize(self.orig_img, (1, self.d))
        print("after orig_img_vec")
        print(orig_img_vec)

        ## w adv image initializization
        w_img_vec = np.arctanh(2 * (orig_img_vec))
        print("after w_img_vec")

        best_adv_img = []
        best_delta = []
        best_distortion = (0.5 * self.d) ** 2
        total_loss = np.zeros(self.I)
        attack_flag = False
        first_flag = True

        #for i in tqdm(range(self.I)):
        for i in range(1):
            ## Total loss evaluation
            total_loss[i] = self.function_evaluation(w_img_vec)
            '''
            ## gradient estimation w.r.t. w_img_vec
            grad_est = gradient_estimation(self.mu, self.q, w_img_vec, self.d, self.kappa, self.target_label, 
                                           self.const, self.model, self.orig_img, self.gradient_descent, 
                                           self.targeted_attack)
            ''' 
        print("end of generate_attack")
            

"""
code = 2
args = {}
if code == 1:
    Imax = 800
    dataset = "mnist"
    args["maxiter"] = Imax + 0
    args["init_const"] = 1
    args["dataset"] = "mnist"
    image_id = 3
    args["img_id"] = img_id
"""
