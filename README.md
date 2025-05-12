This is my implementation of the Vision Transformer project based on the paper "An Image is worth 16x16 words": https://arxiv.org/pdf/2010.11929
My ViT model is in the models folder, and uses Pytorch to implement the Patch Embedding, Transformer and it's components, and MLP head. 
I will test and train my model on the CIFAR100 dataset which can be found in the "dataset" folder. 

To train the transformer, run the following command in the root directory:
python train.py --config config.yaml --load-model 'model path'

To test, run the following command in the root directory:
python test.py --config config.yaml --load-model 'model path'

Current Status: 
Finished Training on 100 epochs on the dataset with model parameters in logs/checkpoint_100.pth.
Training Loss = 0.1621, and testing loss is 2.4870 and testing accuracy is 55.10% 
Results indicate that the model is over fitting the dataset (low training loss, high testing loss). Need either data augmentation, regularization, or model improvements. 


