import torch
from .base_model import BaseModel
from . import networks
import numpy as np
from torch import nn
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import util.util


def Mat(Lvec):
    N = Lvec.size(0)
    Mask = Lvec.repeat(N,1)
    Same = (Mask==Mask.t())
    return Same.clone().fill_diagonal_(0), ~Same#same diff
    
class SCTLoss(nn.Module):
    def __init__(self, method, lam=1):
        super(SCTLoss, self).__init__()
        
        if method=='sct':
            self.sct = True
            self.semi = False
        elif method=='hn':
            self.sct = False
            self.semi = False
        elif method=='shn':
            self.sct = False
            self.semi = True
        else:
            print('loss type is not supported')
            
        self.lam = lam

    def forward(self, fvec, Lvec):
        # number of images
        N = Lvec.size(0)
        
        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)
        
        # Same/Diff label Matting in Similarity Matrix
        Same, Diff = Mat(Lvec.view(-1))
        
        # Similarity Matrix
        CosSim = util.util.fun_CosSim(fvec_norm, fvec_norm)
        
        ############################################
        # finding max similarity on same label pairs
        D_detach_P = CosSim.clone().detach()
        D_detach_P[Diff] = -1
        D_detach_P[D_detach_P>0.9999] = -1
        V_pos, I_pos = D_detach_P.max(1)
 
        # valid positive pairs(prevent pairs with duplicated images)
        Mask_pos_valid = (V_pos>-1)&(V_pos<1)

        # extracting pos score
        Pos = CosSim[torch.arange(0,N), I_pos]
        Pos_log = Pos.clone().detach().cpu()
        
        ############################################
        # finding max similarity on diff label pairs
        D_detach_N = CosSim.clone().detach()
        D_detach_N[Same] = -1
        
        # Masking out non-Semi-Hard Negative
        if self.semi:    
            D_detach_N[(D_detach_N>(V_pos.repeat(N,1).t()))&Diff] = -1
            
        V_neg, I_neg = D_detach_N.max(1)
            
        # valid negative pairs
        Mask_neg_valid = (V_neg>-1)&(V_neg<1)

        # extracting neg score
        Neg = CosSim[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()
        
        # Mask all valid triplets
        Mask_valid = Mask_pos_valid&Mask_neg_valid
        
        # Mask hard/easy triplets
        margin = 0.2
        HardTripletMask = ((Neg>Pos) | (Neg>(1-margin))) & Mask_valid
        EasyTripletMask = ((Neg<Pos) & (Neg<(1-margin))) & Mask_valid
        
        # number of hard triplet
        hn_ratio = (Neg>Pos)[Mask_valid].clone().float().mean().cpu()
        
        # triplets
        Triplet_val = torch.stack([Pos,Neg],1)
        Triplet_idx = torch.stack([I_pos,I_neg],1)
        
        Triplet_val_log = Triplet_val.clone().detach().cpu()
        Triplet_idx_log = Triplet_idx.clone().detach().cpu()
        
        # loss
        if self.sct: # SCT setting
            
            loss_hardtriplet = Neg[HardTripletMask].sum()
            loss_easytriplet = -F.log_softmax(Triplet_val[EasyTripletMask,:]/0.1, dim=1)[:,0].sum()
            
            N_hard = HardTripletMask.float().sum()
            N_easy = EasyTripletMask.float().sum()
            
            if torch.isnan(loss_hardtriplet) or N_hard==0:
                loss_hardtriplet, N_hard = 0, 0
                #print('No hard triplets in the batch')
                
            if torch.isnan(loss_easytriplet) or N_easy==0:
                loss_easytriplet, N_easy = 0, 0
                #print('No easy triplets in the batch')
                
            N = N_easy + N_hard
            if N==0: N=1
            loss = (loss_easytriplet + self.lam*loss_hardtriplet)/N
                
        else: # Standard Triplet Loss setting
            
            loss = -F.log_softmax(Triplet_val[Mask_valid,:]/0.1, dim=1)[:,0].mean()
            
        #print('loss:{:.3f} hn_rt:{:.3f}'.format(loss.item(), hn_ratio.item()), end='\r')

        return loss, hn_ratio

#FTGAN
class FontGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        parser.set_defaults(norm='batch', netG='FTGAN_MLAN', dataset_mode='font')
        if is_train:
            parser.set_defaults(batch_size=64, pool_size=0, gan_mode='hinge', netD='basic_64')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for style loss')
            parser.add_argument('--lambda_content', type=float, default=1.0, help='weight for content loss')
            parser.add_argument('--lambda_SC', type=float, default=0.0, help='weight for L1 loss')
            parser.add_argument('--use_spectral_norm', default=True)
        return parser

    def __init__(self, opt):
        """Initialize the font_translator_gan class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.style_channel = opt.style_channel
        self.sct_style_imgs = []
        self.sct_style_labels = []
        self.sct_batch_size = 512
        self.sct_same_class_picks = 2
        self.batch_size = opt.batch_size
        self.loss_sct = 0
        self.loss_hn_ratio = 0
            
        if self.isTrain:
            self.visual_names = ['gt_images', 'generated_images', 'content_images']+['style_images_{}'.format(i) for i in range(self.style_channel)]
            self.model_names = ['G', 'D_content', 'D_style']
            self.loss_names = ['G_GAN', 'G_L1', 'D_content', 'D_style', 'sct', 'hn_ratio']
        else:
            self.visual_names = ['gt_images', 'generated_images']
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(self.style_channel+1, 1, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD_content = networks.define_D(2, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)
            self.netD_style = networks.define_D(self.style_channel+1, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)
            
            # define loss functions
            self.lambda_L1 = opt.lambda_L1
            self.lambda_SC = opt.lambda_SC
            self.sctloss = SCTLoss('sct')
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.lambda_style = opt.lambda_style
            self.lambda_content = opt.lambda_content
            self.optimizer_D_content = torch.optim.Adam(self.netD_content.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_style = torch.optim.Adam(self.netD_style.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D_content)
            self.optimizers.append(self.optimizer_D_style)
    
    def set_input(self, data):
        self.gt_images = data['gt_images'].to(self.device)
        self.content_images = data['content_images'].to(self.device)
        self.style_images = data['style_images'].to(self.device)
        self.style_indexes = data['style_indexes'].to(self.device)
        if not self.isTrain:
            self.image_paths = data['image_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.generated_images, self.generated_style = self.netG((self.content_images, self.style_images)) 
        
    def compute_gan_loss_D(self, real_images, fake_images, netD):
        # Fake
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real = torch.cat(real_images, 1)
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D
    
    def compute_gan_loss_G(self, fake_images, netD):
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True, True)
        return loss_G_GAN
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        self.loss_D_content = self.compute_gan_loss_D([self.content_images, self.gt_images],  [self.content_images, self.generated_images], self.netD_content)
        self.loss_D_style = self.compute_gan_loss_D([self.style_images, self.gt_images], [self.style_images, self.generated_images], self.netD_style)
        self.loss_D = self.lambda_content*self.loss_D_content + self.lambda_style*self.loss_D_style  
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.loss_G_content = self.compute_gan_loss_G([self.content_images, self.generated_images], self.netD_content)
        self.loss_G_style = self.compute_gan_loss_G([self.style_images, self.generated_images], self.netD_style)
        self.loss_G_GAN = self.lambda_content*self.loss_G_content + self.lambda_style*self.loss_G_style
            
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.generated_images, self.gt_images) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        
    def optimize_parameters_sct(self):
        self.sct_style_imgs = torch.cat(self.sct_style_imgs)
        p = self.sct_same_class_picks
        self.sct_style_labels = torch.cat(self.sct_style_labels)  
        sct_style_labels = [self.sct_style_labels[i//p] for i in range(len(self.sct_style_labels) * p)]
        self.sct_style_labels = torch.stack(sct_style_labels)        

        # Set input
        content_images = self.content_images
        style_images = self.style_images
        self.content_images = None
        self.style_images = self.sct_style_imgs

        self.forward()                   # compute fake images: G(A)
        self.optimizer_G.zero_grad()                  # set G's gradients to zero
        C = self.generated_style.shape[1]     
        self.loss_sct, self.loss_hn_ratio = self.sctloss(self.generated_style.view(-1, C), self.sct_style_labels)
        self.loss_sct *= self.lambda_SC
        if self.loss_sct != 0: # No easy and no hard triplets in batch
            self.loss_sct.backward()
        self.optimizer_G.step()                       # udpate G's weights
        self.sct_style_imgs = []
        self.sct_style_labels = []

        self.content_images = content_images
        self.style_images = style_images

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad([self.netD_content, self.netD_style], True)
        self.optimizer_D_content.zero_grad()
        self.optimizer_D_style.zero_grad()
        self.backward_D()
        self.optimizer_D_content.step()
        self.optimizer_D_style.step()
        # update G
        self.set_requires_grad([self.netD_content, self.netD_style], False)
        self.optimizer_G.zero_grad()                  # set G's gradients to zero
        self.backward_G()                             # calculate graidents for G
        self.optimizer_G.step()                       # udpate G's weights

        if self.lambda_SC > 0:
            self.sct_style_imgs.append(self.style_images)
            self.sct_style_labels.append(self.style_indexes)
            if len(self.sct_style_imgs) * self.sct_same_class_picks * self.batch_size >= self.sct_batch_size:
                self.optimize_parameters_sct()

    def compute_visuals(self):
        if self.isTrain:
            self.netG.eval()
            with torch.no_grad():
                self.forward()
            for i in range(self.style_channel):
                setattr(self, 'style_images_{}'.format(i), torch.unsqueeze(self.style_images[:, i, :, :], 1))
            self.netG.train()
        else:
            pass    