import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils_trans import create_training_dataloader
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, CamembertModel, get_linear_schedule_with_warmup, RobertaConfig, RobertaTokenizer, RobertaModel, LongformerTokenizer, LongformerForSequenceClassification, LongformerConfig



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

class CustomLFForRegression(nn.Module):
    def __init__(self, model_name='allenai/longformer-base-4096', num_labels=1, num_layers_to_freeze=202):
        super(CustomLFForRegression, self).__init__()
        
        # Load Longformer configuration
        config = LongformerConfig.from_pretrained(model_name)
        config.num_labels = 1
        
        # Load pre-trained Longformer model
        self.longf = LongformerForSequenceClassification.from_pretrained(model_name, config=config)
        
        # Iterate through the layers and freeze/unfreeze accordingly
        for i, param in enumerate(self.longf.parameters()):
            if (i <= num_layers_to_freeze) and (i>4):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Modify the model's head for regression
        self.regressor = nn.Linear(config.hidden_size, 1, dtype=torch.float32)  # Assuming BERT's hidden size is 768

        #freeze the first x layers
        for name, param in self.longf.named_parameters():
            print(name, param.requires_grad)


    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        
        #forward pass
        outputs = self.longf(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        logits = outputs.logits
        return logits



# Define input size, hidden layer size, and output size
hidden_channels = 32
hidden_size = 768
output_size = 1


# Instantiate the model, optimizer, and loss function
model = CustomLFForRegression().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.MSELoss()

# Load weights from a .pth file
checkpoint_path = 'LF_lhs.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))  # Specify map_location based on your device

# Load the model state_dict from the checkpoint
model.load_state_dict(checkpoint)

# Define window size and overlap
window_size = 5600
overlap = 100

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

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

        # Split the long text into 500-character increments
        for i in range(0, len(result_string), window_size):
            if np.random.uniform() < .5:
                if len(input_ids) < 5:
                    segment = result_string[i:i+window_size]
                    
                    # Tokenize the segment
                    tokens = tokenizer(segment, padding='max_length', return_tensors='pt')

                    if tokens['input_ids'].size(1) == 4096:
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

        # send tensors to device
        targets = targets.to(dtype=torch.float32).to(device)
        input_ids = torch.cat(input_ids, dim=0).to(device)
        attn_masks = torch.cat(attn_masks, dim=0).to(device)

        #get outputs
        outputs = model(input_ids, attn_masks)
        aggregated_logits = torch.mean(outputs, dim=0, keepdim=True)

        #compute and print loss
        loss = criterion(aggregated_logits, targets)
        loss = loss.to(dtype=torch.float32)
        print(loss)
        outputs_cpu = aggregated_logits.cpu()
        print(aggregated_logits.item())

        loss.backward()

        #optional clip gradients if necessary
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        #accumulate gradients
        if (step + 1) % gradient_accumulation_steps == 0:
            total_loss += loss.item()
            # Update optimizer and scheduler
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            torch.save(model.state_dict(), 'LF_lhs.pth')
        step += 1

    average_loss = total_loss / len(training_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')


    '''
    # Validation loop
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            targets = targets.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits.squeeze(), targets)
            val_loss += loss.item()

    average_val_loss = val_loss / len(validation_loader)
    print(f'Validation Loss: {average_val_loss}')
    '''




