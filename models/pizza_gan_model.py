import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import random

class PizzaGANModel(BaseModel):
    """
    This class implements the PizzaGAN model

    The model training requires '--dataset_mode pizza' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(
            no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0,
                                help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_adv','D_cls', 'G_A','G_B', 'cycle_A','G_adv','reg']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A','A','mask_A', 'fake_B','B','mask_B', 'rec_A']
        #visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        # if self.isTrain and self.opt.lambda_identity > 0.0:
        #     visual_names_A.append('idt_B')
        #     #visual_names_B.append('idt_A')

        # combine visualizations for A and B
        self.visual_names = visual_names_A #+ visual_names_B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
       

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A=[]
        self.netG_B=[]
        self.netG_Amask=[]
        self.netG_Bmask=[]
        if self.isTrain:
            self.model_names += ['G_A', 'G_Amask', 'G_B', 'G_Bmask', 'D', 'Dadv']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_Amask', 'G_B', 'G_Bmask']
        for i in range(opt.num_class):
            tG_A, tG_Amask = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            
            self.netG_A.append(tG_A)
            self.netG_Amask.append(tG_Amask)
            tG_B, tG_Bmask = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B.append(tG_B)
            self.netG_Bmask.append(tG_Bmask)

        self.netD= networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,opt.num_class)
        self.netDadv = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 1)
            

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # create image buffer to store previously generated images
            # self.fake_A_pool = ImagePool(opt.pool_size)
            # create image buffer to store previously generated images
            # self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            # define GAN loss.
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionGAN_D = networks.GANLoss('multi-label').to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizers_G=[]
            for i in range(opt.num_class):
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A[i].parameters(
                ), self.netG_B[i].parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))               
                self.optimizers_G.append(self.optimizer_G)
                
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(
            ), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers=self.optimizers_G+[self.optimizer_D]

    def set_input(self,input,label,orders,big_iter):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A= (input['data'].to(self.device))
        self.image_paths = input['paths']
        self.labels=[]
        self.orders=[int(i)-1 for i in orders]
        self.labels.append(label.copy())
        self.big_iter=big_iter
        for i in range(len(orders)):
            label[self.orders[i]] = 0
            self.labels.append(label.copy())
        self.orders_rev=[]
        self.labels_rev = []
        for i in range(self.opt.num_class):
            if(i not in self.orders):
                self.orders_rev.append(i)
                temp=self.labels[0].copy()
                temp[i]=1
                self.labels_rev.append(temp.copy())
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.orders_rev)
        random.seed(randnum)
        random.shuffle(self.labels_rev)


        
            


    def forward(self,i,direction):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        """the direction argument is used to dertermine the direcrtion of the forward function, designed for the equilibrium of the two classes of the datasets"""
        if(direction):
            self.mask_A = self.netG_Amask[self.orders[i]](self.real_A)
            self.A = self.netG_A[self.orders[i]](self.real_A)
            self.fake_B = self.A.mul(self.mask_A
                )+(1-self.mask_A).mul(self.real_A)  # G_A(A)
            self.mask_B = self.netG_Bmask[self.orders[i]](self.fake_B)
            self.B = self.netG_B[self.orders[i]](self.fake_B)
            self.rec_A = self.B.mul(self.mask_B)+(1-self.mask_B).mul(self.fake_B)   # G_B(G_A(A))
        else:
            self.mask_A = self.netG_Bmask[self.orders_rev[i]](self.real_A)
            self.A = self.netG_B[self.orders_rev[i]](self.real_A)
            self.fake_B = self.A.mul(self.mask_A
                                     )+(1-self.mask_A).mul(self.real_A)  # G_A(A)
            self.mask_B = self.netG_Amask[self.orders_rev[i]](self.fake_B)
            self.B = self.netG_A[self.orders_rev[i]](self.fake_B)
            self.rec_A = self.B.mul(
                self.mask_B)+(self.mask_B).mul(1-self.fake_B)   # G_B(G_A(A))


    def backward_D_basic(self, netD, netDadv,real,fake,i,direction):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # if(not self.orders):
        #     pred_real = netD(real)      
        #     loss_D_real = self.criterionGAN_D(pred_real, self.labels[i])
        #     # Fake
        #     pred_fake = netD(fake.detach())
        #     loss_D_fake = self.criterionGAN_D(pred_fake, self.labels[i+1])
        #     # Combined loss and calculate gradients
        #     self.loss_D_cls = (loss_D_real + loss_D_fake) * 0.5
        # else:
        if(direction):
            self.pred_real = netD(real)
            self.loss_D_cls = self.criterionGAN_D(self.pred_real, self.labels[i])
            ifvalidAdorn=netDadv(real)
            ifvalidNoAdorn=netDadv(fake.detach())
            loss_D_adv_real = self.criterionGAN_D(ifvalidAdorn,True)
            loss_D_adv_fake = self.criterionGAN_D(ifvalidNoAdorn,False)
            self.loss_D_adv=(loss_D_adv_fake+loss_D_adv_real)*0.5
        else:
            pred_real = netD(real)
            self.loss_D_cls = self.criterionGAN_D(
                pred_real, self.labels_rev[i])
            ifvalidAdorn=netDadv(real)
            ifvalidNoAdorn=netDadv(fake.detach())
            loss_D_adv_real = self.criterionGAN_D(ifvalidAdorn,True)
            loss_D_adv_fake = self.criterionGAN_D(ifvalidNoAdorn,False)
            self.loss_D_adv=(loss_D_adv_fake+loss_D_adv_real)*0.5

        loss_D = self.loss_D_cls+self.loss_D_adv
        loss_D.backward()
        return loss_D

    def backward_D(self,i,direction):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D = self.backward_D_basic(self.netD,self.netDadv,self.real_A,self.fake_B,i,direction)
        
    def backward_DX(self):
        pred = self.netD(self.real_A)
        self.loss_D_cls = self.criterionGAN_D(pred, torch.zeros_like(pred))
        loss_D=self.loss_D_cls
        loss_D.backward()
        return loss_D
    def backward_DY(self):
        pred=self.netD(self.real_A)
        self.loss_D_cls = self.criterionGAN_D(pred, torch.ones_like(pred))
        loss_D=self.loss_D_cls
        loss_D.backward()
        return loss_D

    # def backward_D_B(self):
    #     """Calculate GAN loss for discriminator D_B"""
    #     fake_A = self.fake_A_pool.query(self.fake_A)
    #     self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self,i,direction):
        """Calculate the loss for generators G_A and G_B"""
        #lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_reg = 0.01
        # Identity loss
        if(direction):
            #the idt loss needs to be removed"""

            # if lambda_idt > 0:
            #     # G_A should be identity if real_B is fed: ||G_A(B) - B||   使用fakeB代替
            #     self.idt_A = self.netG_A[self.orders[i]](self.fake_B)
            #     self.loss_idt_A = self.criterionIdt(
            #         self.idt_A, self.fake_B) * lambda_B * lambda_idt
            #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
            #     self.idt_B = self.netG_B[self.orders[i]](self.real_A)
            #     self.loss_idt_B = self.criterionIdt(
            #         self.idt_B, self.real_A) * lambda_A * lambda_idt
            # else:
            #     self.loss_idt_A = 0
            #     self.loss_idt_B = 0
            self.loss_G_adv=self.criterionGAN_D(self.netDadv(self.fake_B),True)
            # GAN loss D_A(G_A(A))
            self.pred_fake = self.netD(self.fake_B)
            self.loss_G_A = self.criterionGAN_D(self.pred_fake,self.labels[i+1])
            # GAN loss D_B(G_B(B))
            
            self.loss_G_B = self.criterionGAN_D(self.netD(self.rec_A), self.labels[i])
            
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            #self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
            self.criterionReg=torch.nn.MSELoss()
            #
            self.loss_reg = (self.criterionReg(self.mask_A, torch.ones_like(self.mask_A))+self.criterionReg(self.mask_B, torch.ones_like(self.mask_B)))*0.5*lambda_reg
            # combined loss and calculate gradients
            self.loss_G = self.loss_G_adv+self.loss_G_A + self.loss_cycle_A + self.loss_G_B
            self.loss_G.backward()
        else:
            # if lambda_idt > 0:
            #     # G_A should be identity if real_B is fed: ||G_A(B) - B||   使用fakeB代替
            #     self.idt_A = self.netG_B[self.orders_rev[i]](self.fake_B)
            #     self.loss_idt_A = self.criterionIdt(
            #         self.idt_A, self.fake_B) * lambda_B * lambda_idt
            #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
            #     self.idt_B = self.netG_A[self.orders_rev[i]](self.real_A)
            #     self.loss_idt_B = self.criterionIdt(
            #         self.idt_B, self.real_A) * lambda_A * lambda_idt
            # else:
            #     self.loss_idt_A = 0
            #     self.loss_idt_B = 0
            self.loss_G_adv = self.criterionGAN_D(self.netDadv(self.fake_B), True)
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN_D(
                self.netD(self.fake_B), self.labels_rev[i])
            # GAN loss D_B(G_B(B))

            self.loss_G_B = self.criterionGAN_D(
                self.netD(self.rec_A), self.labels[0])

            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(
                self.rec_A, self.real_A) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            #self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
            self.criterionReg = torch.nn.MSELoss()
            self.loss_reg = -(self.criterionReg(self.mask_A, torch.ones_like(self.mask_A)) +
                              self.criterionReg(self.mask_B, torch.ones_like(self.mask_B)))*0.5*lambda_reg
            # combined loss and calculate gradients
            self.loss_G = self.loss_G_adv+self.loss_G_A + self.loss_cycle_A +self.loss_G_B
            self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        for i in range(min(self.big_iter+1,len(self.orders_rev))):
            if(self.orders_rev):
                # compute fake images and reconstruction images.
                self.forward(i,False)
                # G_A and G_B
                # Ds require no gradients when optimizing Gs
                self.set_requires_grad(self.netD, False)
                # set G_A and G_B's gradients to zero
                self.optimizers_G[self.orders_rev[i]].zero_grad()
                # calculate gradients for G_A and G_B
                self.backward_G(i,False)
                # update G_A and G_B's weights
                self.optimizers_G[self.orders_rev[i]].step()
                # D_A and D_B
                self.set_requires_grad(self.netD, True)
                self.optimizer_D.zero_grad()  
                self.backward_D(i,False)      
                self.optimizer_D.step() 
            else:
                self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
                self.backward_DY()      # calculate gradients for D_A
                self.optimizer_D.step()
        for i in range(min(self.big_iter+1, len(self.orders))):
            if(self.orders):
                if(i>0):
                    self.real_A = self.fake_B.detach()
                self.forward(i,True)      # compute fake images and reconstruction images.
                # G_A and G_B
                # Ds require no gradients when optimizing Gs
                self.set_requires_grad(self.netD, False)
                # set G_A and G_B's gradients to zero
                self.optimizers_G[self.orders[i]].zero_grad()
                self.backward_G(i,True)             # calculate gradients for G_A and G_B
                # update G_A and G_B's weights
                self.optimizers_G[self.orders[i]].step()
                # D_A and D_B
                self.set_requires_grad(self.netD, True)
                self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
                self.backward_D(i,True)      # calculate gradients for D_A
                self.optimizer_D.step()            
            else:
                self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
                self.backward_DX()      # calculate gradients for D_A
                self.optimizer_D.step()  
            self.current_label=self.labels[0]
            self.current_order=self.orders
            self.current_pred = np.concatenate((self.pred_real.detach().cpu().numpy().mean(
                axis=2).mean(axis=2), self.pred_fake.detach().cpu().numpy().mean(axis=2).mean(axis=2)))





        
        
