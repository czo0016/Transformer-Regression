# Transformer-Regression

Tranformer model used for regression in predicting essay scores from keystrokes.
Data from https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality 

# Overview of Data and Regression Goal

The training data provided by the organizers comes from logs of typing actions that participants took while writing an essay, for roughly 3000 essays. Each row of the training data is a single action, and includes, among other fields:
• An identifier for the essay it belongs to
• A millisecond timestamp
• The type of action (backspace, character, cut, paste, etc.) 
• Cursor position

The target variables are the scores each essay received, from 1-6, when graded by a human. The goal of the task is to train a model that is able to predict the score of the essay based on the actions. The organizers are explicit that a root-mean-squared error will be used to evaluate the model, using the root-mean-squared difference between the predicted score y ̃ and the correct score y for each row in the test set. The real test set is hidden by the organizers until the competition is complete.

# Data Proprocessing
For the transformer data preprocessing can be found in utils_trans.py
Ultimately each essay key logs were changed to a single string. An abbreviated example below:

  "[1 31 Nonproduction Leftclick Leftclick NoChange 0 0 2 404 Input Shift Shift NoChange 0 405...]"

# Model Modification

Two type of Transformer networks were used, RoBERTa and Longformer. These networks were used due to their prevalance, and Longformer's ability to handle long sequence lengths (2048 tokens).

Both networks were modified to change the activation function to identity and passed to a fully connected linear layer. This allowed for the prediction of an essay score (1-6) as a regression problem rather than classification. Additionally, both networks had their first 11 layers frozen and only the paramters of the lower layers updated on training.

# Challenges

Hardware constraints were a major problem with this type of network. It would be best to run this model on a dedicated GPU. Due to this each essay was split into batches of tokens and only 25 batches were used to predict the score. The 25 batches were chosen at random. Then the average of the batches was compared to the target score to obtain the loss. Additionally since hardware was so constrained, batch training was not possible, so gradients had to be accumulated and then backpropogated every 25 samples.

# Results

After 1.5 epochs with a special focus on high and low essay scores, the results were as follows for RoBERTa:
Training: RMSE 0.671 Validation: RMSE 0.867

Longformer:
Training: RMSE 0.652 Validation: RMSE 0.851

This is significantly less than the leaderboard (Validation RMSE: .559)

However, I suspect with additional training on ML dedicated hardware the error would improve significantly. Additionally, this was an exercise to apply deep learning techniques to unstructured data, rather than take the typical route of feature creation + classical ML. The goal was to learn the process rather than compete for the best results.



