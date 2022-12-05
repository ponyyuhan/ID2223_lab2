# ID2223_lab2

We use Google Drive to save the dataset (arrow files) due to its need for high storage capacity. 

Huggingface link(Allow the user to paste in the URL to a video, and transcribe what is
spoken in the video):https://huggingface.co/spaces/Campfireman/whisper_lab2


# Model-centric optimization
1. num_train_epochs: Considering the calculation power limit and possible overfitting problems, we keep the num_train_epochs to be 1 as default. The number of samples is large enough to cover almost all the language usages and there is no need to fit the model for a second time. 
2. per_device_train_batch_size and gradient_accumulation_steps: The product of these two parameters defines how many samples are fed into the model per step in gradient change. Less batch data per step may introduce more fluctuations in node parameters, while with more batch data the gradient may change in a slower manner. In this project, we use 16 * 1 in training process. 
3. learning_rate: A smaller starting learning rate results in a rather precise aproach towards the goal. In this project we use 1e-5 as the initial rate. 
4. per_device_eval_batch_size: A higher value can accelerate the throughput of evaluation. Due to the limitation of calculation power, this parameter is set to 8. 
5. predict_with_generate: This is a commonly used language precision metric set and we enabled it here as this project deals with natural language process.

