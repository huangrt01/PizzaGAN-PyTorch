
<br><br><br>

# PizzaGAN-inofficial-incomplete in PyTorch
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
epoch 22 results

I am uncertain about whether the model needs to be modified to generate the convincing results or it just needs more training to make the generator remove the toppings on the pizza thoroughly, hoping anyone who sees this program can help solve this problem.
<img src="https://raw.githubusercontent.com/huangrt01/PizzaGAN-inofficial-incomplete/master/imgs/epoch22img.jpg" width="800"/>
<img src="https://raw.githubusercontent.com/huangrt01/PizzaGAN-inofficial-incomplete/master/imgs/epoch22img4.jpg" width="800"/>



## Acknowledgments
code is based on [pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
