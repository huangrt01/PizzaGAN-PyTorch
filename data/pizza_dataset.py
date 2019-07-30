import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random


class pizzaDataset(BaseDataset):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     """Add new dataset-specific options, and rewrite default values for existing options.

    #     Parameters:
    #         parser          -- original option parser
    #         is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

    #     Returns:
    #         the modified parser.
    #     """
    @staticmethod
    def modify_commandline_options(parser,is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--num_class', type=int,
                            default=10, help='number of the classes')
        return parser


    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

  
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        randnum = random.randint(0, 100)
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.dir = os.path.join(opt.dataroot,opt.phase,'images')  
        self.image_paths=sorted(make_dataset(self.dir,opt.max_dataset_size))
   
        random.seed(randnum)
        random.shuffle(self.image_paths)
        
        self.size=len(self.image_paths)
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        self.transform = get_transform(opt,grayscale=(input_nc==1))
        dir_order=os.path.join(opt.dataroot,opt.phase,opt.phase+'Order.txt')
        dir_labels = os.path.join(opt.dataroot, opt.phase, opt.phase+'Labels.txt')
        self.labels=(np.loadtxt(dir_labels)).tolist()

        random.seed(randnum)
        random.shuffle(self.labels)

        self.labels=np.array(self.labels)

        self.order=[]
        with open(dir_order) as f:  # 每次读⼀⾏
            for line in f:
                fileds = line.split()
                row_data = [float(x) for x in fileds] 
                self.order.append(row_data[::-1])

        random.seed(randnum)
        random.shuffle(self.order)

        self.order = np.array(self.order)
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index % self.size]  # make sure index is within then range
        img = Image.open(path).convert('RGB')
        # apply image transformation
        img = self.transform(img)
        return {'data': img, 'paths': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
