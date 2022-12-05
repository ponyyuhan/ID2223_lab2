# ID2223_lab2

We use Google Drive to save the dataset (arrow files) due to its need for high storage capacity. 


# Model-centric optimization
1. num_train_epochs: Considering the calculation power limit and possible overfitting problems, we keep the num_train_epochs to be 1 as default. The number of samples is large enough to cover almost all the language usages and there is no need to fit the model for a second time. 
2. per_device_train_batch_size and gradient_accumulation_steps: The product of these two parameters defines how many samples are fed into the model per step in gradient change. Less batch data per step may introduce more fluctuations in model parameters, while with more batch data the gradient may change in a slower manner. 
3. 
