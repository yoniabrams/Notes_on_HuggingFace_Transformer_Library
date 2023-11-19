# Notes on using Hugging Face's Transformer Library

Thanks to the folks out at **[AssemblyAI](https://www.assemblyai.com/)** for the incredible tutorials, particularly the one which inspired the structure and much of the content of the below. Find that video [here](https://www.youtube.com/watch?v=QEaBAZQCtwE) and the rest of their videos [here](https://www.youtube.com/@AssemblyAI).

## Installation
It's simple; type the following into your command line
```commandline
pip install transformers 
```

## Pipelines
Pipelines help streamline machine learning tasks

Hugging Face's Transformer Library has its own native implementation of pipelines:
```python
from transformers import pipeline
```
### Examples:
Here's an example of using a pipeline using the default model for a `sentiment-analysis` task.
- Note: If you don't specify a model, then Hugging Face will pick a default model for the given task.
```python
task = 'sentiment-analysis'

classifier = pipeline(task=task)

result = classifier('This is an incredible README which explains how to use the Hugging Face Transformer Library!')
```

Here's another example where we pick a particular  model for a `text-generation` task:
- You can either use a model which you've saved **locally**, or a model from the Hugging Face **model hub** (as in this example).
```python
task = 'text-generation'
model = 'distilgpt2'

generator = pipeline(task=task, model=model)

results = generator(
  'In this README, we are sharing notes on how to',
  max_length=30,  # this argument tells the pipeline to output a text up to this # of characters
  num_return_sequences=2  # this argument tells the pipeline to output 2 results 
)

print(results)
```

Here's a third example using `zero-shot-classification`:
```python
task = 'zero-shot-classification'

classifier = pipeline(task=task)

result = classifier(
  ['This is the text which we will attempt to classify',
   candidate_labels=['education', 'politics', 'machine learning']]
)

print(result)
```

### What does a pipleine do for us? 
A pipeline streamlines **3 steps** for us:
1. **Preprocessing** --> It preprocesses the text by applying a `tokenizer` to the text.
2. **Prediction** --> Feeds preprocessed text to the model.
3. **Postprocessing** --> The pipeline shows us a result as we would expect it. For example, in a `sentiment-analysis` pipeline, the post-processed result of the model would look like this:
```python
# continuing the example from above...
print(result)

# Output:
# [{'label': 'POSITIVE', 'score': 0.9598047358505}]
```

### Supported Tasks:
Hugging Face Transformer.pipelines supports the following tasks (as of 19-11-23), among others.

*(You can find the updated documentation [here](https://github.com/huggingface/transformers/blob/71688a8889c4df7dd6d90a65d895ccf4e33a1a56/src/transformers/pipelines.py#L2716-L2804).)*
1. `'feature-extraction'`
- The process of transforming raw data into numerical features that can be processed while preserving the information.
<br><br>
2. `'sentiment-analysis'`
- The process of assessing the sentiment (POSITIVE or NEGATIVE) from a text. 
<br><br>
3. `'ner'`
- NER = Named Entity Recognition. NER is the process of locating and classifying named entities (types of pre-defined categories) in unstructured text. Example categories include:
  - Names of people, organization names, locations, medical codes, etc. 
<br><br>
4. `'question-answering'`
- These models can retrieve the answer to a question from a given text.
<br><br>
5. `'fill-mask'`
- The process of predicting the words which should replace the mask which were left unfilled in a given text. 
- The following is an example of a text with a mask from Hugging Face's page on [Fill-Mask](https://huggingface.co/tasks/fill-mask):
```python
task = 'fill-mask'
classifier = pipeline(task)
classifier('Paris is the <mask> of France.')

# [{'score': 0.7, 'sequence': 'Paris is the capital of France.'},
# {'score': 0.2, 'sequence': 'Paris is the birthplace of France.'},
# {'score': 0.1, 'sequence': 'Paris is the heart of France.'}]
```
<br><br>
6. `'summarization'`
- The process of summarizing a given text.
<br><br>
7. `'translation'`
- The process of translating a text.
<br><br>
8. `'text2text-generation'`
- The process of generating text from a given textual input. This type of model knows how to map from one given text to an output text. 
<br><br>
9. `'text-generation'`
- The process of generating text from a prompt. This isn't for mapping, but for generating text which begins with the given text. 
<br><br>
10. `'zero-shot-classification'`
- The process by which a model can classify data into multiple classes without any specific training examples for those classes.
- An example would help here. This example is taken from Hugging Face's page on [Zero-Shot Classification](https://huggingface.co/tasks/zero-shot-classification):
```python
task = 'zero-shot-classification'

pipe = pipeline(task) # e.g. model="facebook/bart-large-mnli"
pipe("I have a problem with my iphone that needs to be resolved asap!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)
# output
# {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!',
# 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'],
# 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}
```
<br><br>
11. `'conversational'`
- The task of generating conversational text which is (1) relevant, (2) coherent, and (3) knowledgeable, given a prompt. Think ChatGPT...

## Tokenizers and Models
When we create a generic pipeline in the Transformers library, we don't see behind the scenes of what our code is doing. 

For example, the following two codes are equivalent:
1. Without specifying Models or Tokenizers
```python
from transformers import pipeline

task = 'sentiment-analysis'

classifier = pipeline(task=task)

result = classifier('Please, can we get started on the classification already?!')
print(result)
```

2. Specifying Models and Tokenizers
```python
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassiffication

task = 'sentiment-analysis'
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'  # the default model

model = AutoModelForSequenceClassiffication.from_pretrained(model_name) 
tokenizer = AutoTokenizer.from_pretrained(model_name)  # the generic tokenizer

classifier = pipeline(
  task=task,
  model=model,
  tokenizer=tokenizer
)

result = classifier('Please, can we get started on the classification already?!')
print(result)
```

The latter code shows us a deeper insight into what is happening under the hood of a pipeline. 

### What does a tokenizer do?
Let's look at some examples:
```python
sequence = 'This is a sequence of characters using latin, lowercase letters'

# The full tokenized sequence:
result = tokenizer(sequence)
print(result)  # {'input_ids' : [ < List of ids >], 'attention_mask': [ < list of attention masks | i.e. 0s or 1s > ]}

# The tokens of the sequence (e.g. individual words, or parts of words and punctuation):
tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['this', 'is', 'a', 'sequence', 'of', 'characters', 'using', 'latin', ',', 'lowercase', 'letters']

# The ids associated with each token:
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)  # [ < list of numbers > ]

# You can also decode the ids and return them to a sequence:
decoded_string = tokenizer.decode(ids)
print(decoded_string)  # this is a sequence of characters using latin, lowercase letters
```

## Integration with PyTorch and TensorFlow

** PyTorch example:**
```python
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassiffication
import torch
import torch.nn.functional as F

task = 'sentiment-analysis'
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'  # the default model

model = AutoModelForSequenceClassiffication.from_pretrained(model_name) 
tokenizer = AutoTokenizer.from_pretrained(model_name)  # the generic tokenizer

classifier = pipeline(
  task=task,
  model=model,
  tokenizer=tokenizer
)

X_train = [
  'I am really glad someone took readable notes on using Hugging Face',
  'Thank you whomever did this'
]

batch = tokenizer(
  X_train,
  padding=True,  # helps make all the sequences the same minimum length
  truncation=True,  # helps make all the sequences the same maximum length
  max_length=512,  # defining the maximum length
  return_tensor='pt'  # PyTorch | This helps with a smooth integration with PyTorch
)  # batch is now a dictionary

# The following code is how we conduct the inference step of the pipeline in PyTorch
with torch.no_grad():
  outputs = model(**batch)  # we have to 'unpack' the batch of all of its arguments. 
  print(outputs)
  predictions = F.softmax(outputs.logits, dim=1)
  print(predictions)
  labels = torch.argmax(predictions, dim=1)
  print(labels)
```

**TensorFlow Example:**
```python
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

task = 'sentiment-analysis'
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

X_train = [
    'I am really glad someone took readable notes on using Hugging Face',
    'Thank you whomever did this'
]

batch = tokenizer(
    X_train,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='tf'  # TensorFlow | This returns TensorFlow tensors
)

# Implementation using TensorFlow functional API
outputs = model(**batch)
logits = outputs.logits
predictions = tf.nn.softmax(logits, axis=1)
labels = tf.argmax(predictions, axis=1)

print(logits)
print(predictions)
print(labels)
```

## Save and Load Models and Tokenizer

### Saving...
This is relatively simple.
```python
directory = 'saved_here'
tokenizer.save_pretrained(directory)
model.save_pretrained(directory)
```

### Loading...
Loading a saved model or tokenizer is simple too!
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
directory = 'saved_here'
tokenizer = AutoTokenizer.from_pretrained(directory)
model = AutoModelForSequenceClassification.from_pretrained(directory)
```

*Done!*

## Model Hub
You can search for models on the Hugging Face [homepage](https://huggingface.co/)

**Happy searching, tokenizing and modeling!!**

## Finetuning your own model using Hugging Face

Steps for finetuning a model:
1. Prepare your dataset. 
- When we speak about finetuning a model, we usually mean  that you have a dataset for which you'd like to tune the model's performance.
- Therefore, you need to ensure that the dataset is prepared. 

2. Load a pretrained tokenizer and tokenize your dataset to get the encodings

3. If you're using PyTorch, prepare the PyTorch dataset with the encodings from above.

4. Load the pretrained model.

5. Finetune:
   1. Either: Load the `Trainer` and `TrainerArguments` classes from the `transformers` library
   2. Or: Use a native PyTorch training loop

**Finetuning using Trainer and TrainingArguments**<br>
- According to the Hugging Face [documentation](https://huggingface.co/docs/transformers/main_classes/trainer), you have to create a `TrainingArguments` object before instantiating your `Trainer` as it allows you to access all the points of customization during training. Meaning, if you want to customize your training when you implement it with `Trainer` you have to be able to access the customizations in the first place.

Let's look at an implementation of this:
```python
from transformers import Trainer, TrainingArguments

# Assuming you have the necessary datasets, model, tokenizer, and data_collator defined

training_args = TrainingArguments('test-trainer')

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()
```

**Native PyTorch training loop for Finetuning:**
```python
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

# Assuming you have the necessary datasets, model, tokenizer, and data_collator defined
model = ...

train_dataloader = ...
val_dataloader = ...

num_epochs = 3  # obviously this is adjustable as needed

# Define your optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_scheduler(
    "linear",
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * num_epochs
)

# Define your loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3  # You can adjust this as needed

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_dataloader:
        inputs = batch["input_ids"]
        labels = batch["labels"]

        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss}")

# Optionally, evaluate on the validation set
model.eval()
val_loss = 0.0

with torch.no_grad():
    for batch in val_dataloader:
        inputs = batch["input_ids"]
        labels = batch["labels"]

        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        val_loss += loss.item()

avg_val_loss = val_loss / len(val_dataloader)
print(f"Average Validation Loss: {avg_val_loss}")

```