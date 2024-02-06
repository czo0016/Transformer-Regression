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


