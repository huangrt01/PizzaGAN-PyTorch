
<br><br><br>

# PizzaGAN-inofficial in PyTorch
## Datasets
the datasets in on the [official page](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) of the author of PizzaGAN. Please save the [datasets](http://pizzagan.csail.mit.edu/pizzaGANsyntheticdata.zip) in ./datasets/ .

## PizzaGAN Train
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train
```bash
cd PizzaGAN-inofficial-incomplete

python train.py --dataroot ./datasets/syntheticDataset --name pizzapizza --model pizza_gan --gpu_ids 1,2 --batch_size 1 --wizard_batch_size 2 --dataset_mode pizza --epoch_count 0 --serial_batch 
```
- Continue training
```bash
python train.py --dataroot ./datasets/syntheticDataset --name pizzapizza --model pizza_gan --gpu_ids 1,2 --batch_size 1 --wizard_batch_size 2 --dataset_mode pizza --continue_train --epoch 20 --epoch_count 21 --serial_batch 
```



the training doesn't support multiple GPUs training because for every image the training procedure has subtle differences.

## results
the results are in /checkpoints/pizzapizza, using the HTML file to show the results of every epoch.

epoch 11 results:
<img src="https://raw.githubusercontent.com/huangrt01/PizzaGAN-inofficial-incomplete/master/imgs/epoch11.jpg" width="800"/>

<img src="https://raw.githubusercontent.com/huangrt01/PizzaGAN-inofficial-incomplete/master/imgs/epoch11-1.jpg" width="800"/>
<img src="https://raw.githubusercontent.com/huangrt01/PizzaGAN-inofficial-incomplete/master/imgs/epoch11-2.jpg" width="800"/>



## Acknowledgments
code is based on [pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
