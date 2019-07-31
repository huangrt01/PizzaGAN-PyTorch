
<br><br><br>

# PizzaGAN-inofficial-incomplete in PyTorch
## Datasets
the datasets in on the [official page](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) of the author of PizzaGAN. Please save the [datasets](http://pizzagan.csail.mit.edu/pizzaGANsyntheticdata.zip) in ./datasets/ .

## Train
cd PizzaGAN-inofficial-incomplete

python train.py --dataroot ./datasets/syntheticDataset --name pizzapizza --model pizza_gan --gpu_ids 1,2 --batch_size 1 --wizard_batch_size 2 --dataset_mode pizza --epoch 1 --epoch_count 1 --serial_batch 

the training doesn't support multiple GPUs training because for every image the train procedure has slight differences.


## Acknowledgments
code is based on [pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
