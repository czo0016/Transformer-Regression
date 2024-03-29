import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils_trans import create_training_dataloader
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, CamembertModel, get_linear_schedule_with_warmup, RobertaConfig, RobertaTokenizer, RobertaModel



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)

batch_size = 1
training_loader, validation_loader, sample_size = create_training_dataloader(batch_size=batch_size)

class CustomRobertaForRegression(nn.Module):
    def __init__(self, model_name='roberta-base', num_labels=1, num_layers_to_freeze=164):
        super(CustomRobertaForRegression, self).__init__()
        
        # Load RoBERTa configuration
        config = RobertaConfig.from_pretrained(model_name)

        # Load pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained(model_name, config=config)
        
        # Iterate through the layers and freeze/unfreeze accordingly
        for i, param in enumerate(self.roberta.parameters()):
            if (i <= num_layers_to_freeze) and (i>5):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Modify the model's head for regression
        self.roberta.config.num_labels = num_labels

        #self.roberta.pooler.dense = nn.Linear(config.hidden_size, num_labels)
        self.regressor = nn.Linear(config.hidden_size, 1, dtype=torch.float32)  # Assuming BERT's hidden size is 768
        self.roberta.pooler.activation = nn.Identity()

        for name, param in self.roberta.named_parameters():
            print(name, param.requires_grad)


    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        lhs_output = outputs.last_hidden_state
        logits = self.regressor(lhs_output)
        return logits


# Define a simple regression model using BERT as a feature extractor
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.regressor = nn.Linear(hidden_size, 1, dtype=torch.float32)  # Assuming BERT's hidden size is 768

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.regressor(pooled_output)
        return logits

# Define input size, hidden layer size, and output size
hidden_channels = 32
hidden_size = 768
output_size = 1


# Instantiate the model, optimizer, and loss function
model = CustomRobertaForRegression().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.MSELoss()

# Load weights from a .pth file
checkpoint_path = 'RoBERTa_lhs.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))  # Specify map_location based on your device

# Load the model state_dict from the checkpoint
model.load_state_dict(checkpoint)


# Define window size and overlap
window_size = 700
overlap = 100

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')



epochs = 1

# Define gradient accumulation steps
gradient_accumulation_steps = 25


for epoch in range(epochs):
    model.train()
    total_loss = 0
    step = 0
    for batch in training_loader:
        input_ids, targets = batch

        # Concatenate values across columns with spaces
        result_string = ' '.join([' '.join(map(str, arr)) for arr in input_ids])
        
        characters_to_remove = '\[]"\''
        translation_table = str.maketrans("", "", characters_to_remove)
        result_string = result_string.translate(translation_table)

        # Initialize a list to store tokenized results
        input_ids = []
        attn_masks = []

        # Split the long text into 500-character increments, accumlate a random sample from each essay
        for i in range(0, len(result_string), window_size):
            if np.random.uniform() < .15:
                if len(input_ids) < 25:
                    segment = result_string[i:i+window_size]
                    
                    # Tokenize the segment
                    tokens = tokenizer(segment, padding='max_length', return_tensors='pt')

                    if tokens['input_ids'].size(1) == 512:
                    # Append the tokenized result to the list
                        input_ids.append(tokens['input_ids'])
                        attn_masks.append(tokens['attention_mask'])
                else:
                    next
            else:
                next
        if len(input_ids) < 1:
            segment = result_string[i:i+window_size]
            
            # Tokenize the segment
            tokens = tokenizer(segment, padding='max_length', return_tensors='pt')
        
            # Append the tokenized result to the list
            input_ids.append(tokens['input_ids'])
            attn_masks.append(tokens['attention_mask'])


        #targets = targets.repeat(len(input_ids), 1).to(dtype=torch.float32).to(device)
        targets = targets.to(dtype=torch.float32).to(device)
        input_ids = torch.cat(input_ids, dim=0).to(device)
        attn_masks = torch.cat(attn_masks, dim=0).to(device)

        input_ids = input_ids
        attn_masks = attn_masks

        #get outputs
        outputs = model(input_ids, attn_masks)
        aggregated_logits = torch.mean(outputs, dim=1, keepdim=False)
        aggregated_logits = torch.mean(aggregated_logits, dim=0, keepdim=True)

        #caculate loss
        loss = criterion(aggregated_logits, targets)
        loss = loss.to(dtype=torch.float32)
        print(loss)
        outputs_cpu = aggregated_logits.cpu()
        print(aggregated_logits.item())

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        #accumulate gradients and perform gradient descent
        if (step + 1) % gradient_accumulation_steps == 0:
            total_loss += loss.item()
            # Update optimizer and scheduler
            optimizer.step()
            optimizer.zero_grad()
            #scheduler.step()
            torch.save(model.state_dict(), 'RoBERTa_lhs.pth')
        step += 1

    average_loss = total_loss / len(training_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')





'''
#Validation Loop

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    running_loss = 0
    for images, labels in training_loader:


        # move to the specified device
        images, labels = images.float().to(device), labels.float().to(device)

        # forward
        optimizer.zero_grad()
        outputs = model(images)
        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    # statistics
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / sample_size

    print(f"Training Loss: {epoch_loss:.4f}")


    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0
    with torch.no_grad():  # Disable gradient computation during validation
        for val_images, val_labels in validation_loader:
            val_images, val_labels = val_images.float().to(device), val_labels.float().to(device)

            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)

            val_running_loss += val_loss.item() * val_images.size(0)

    val_epoch_loss = val_running_loss / len(validation_loader.dataset)
    
    print(f"Validation Loss: {val_epoch_loss:.4f}")
    print("-" * 10)
'''