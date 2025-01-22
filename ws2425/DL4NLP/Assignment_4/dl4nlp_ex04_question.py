# %% [markdown]
# # Deep Learning for NLP - Exercise 04
# In this exercise, we will first look at the inner workings of the attention mechanism and the importance of various heads in pretrained models. The second task compares multiple decoding strategies on the same model and prompt to highlight how much on an effect the decoding algorithm has on the quality of generated text.
# 
# Task 1 and Task 2 can be worked in independently.
# ___
# 
# General hints:
# * Have a look at the imports below when solving the tasks
# * Use the given modules and all submodules of the imports, but don't import anything else!
#     * For instance, you can use other functions under the `torch` or `nn` namespace, but don't import e.g. PyTorch Lightning, etc.
# * It is recommended to install all packages from the provided environment file
# * Feel free to test your code between sub-tasks of the exercise sheet, so that you can spot mistakes early (wrong shapes, impossible numbers, NaNs, ...)
# * Just keep in mind that your final submission should be compliant to the provided initial format of this file
# 
# Submission guidelines:
# * Make sure that the code runs on package versions from the the provided environment file
# * Do not add or change any imports (also don't change the naming of imports, e.g. `torch.nn.functional as f`)
# * Remove your personal, additional code testings and experiments throughout the notebook
# * Do not change the class, function or naming structure as we will run tests on the given names
# * Additionally export this notebook as a `.py` file, and submit **both** the executed `.ipynb` notebook with plots in it **and** the `.py` file
# * **Deviation from the above guidelines will result in partial or full loss of points**

# %%
# !pip install transformers==4.24.0
# !pip install datasets==3.0.1
# !pip install bertviz==1.4.0
# !pip install plotly==5.17.0

# %% [markdown]
# # Task 1: Looking 'Inside' the Models

# %%
import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt
from bertviz import head_view, model_view

import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    BertConfig,
)

# %% [markdown]
# ## Task 1.1: Visualizing and Analyzing Attention Maps

# %% [markdown]
# * In the following experiment, we compare three models: a randomly initialized BERT model, and trained BERT model, and a trained GPT-2 model
# * Start by loading a randomly initialized BERT model
#     * You can achieve this by loading an [AutoModelForSequenceClassification.from_config](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification.from_config)
#     * Make use of the imported `BertConfig`
#     * Specify `output_attentions=True`
# * Repeat the process for pre-trained BERT and GPT-2
#     * For pre-trained BERT, load [bhadresh-savani/bert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion) using [AutoModelForSequenceClassification.from_pretrained](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification.from_pretrained)
#     * For GPT-2, it is enough to use [AutoModel.from_pretrained](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel.from_config) method
#         * This model will not have a language modelling head ontop, but only return hidden states. Since we are only interested in attention outputs, this is enough in this case
#     * Also specify `output_attentions=True` for both
# * Set all models into `eval()` mode
# * Load BERT's and GPT2's tokenizer accordingly using the `AutoTokenizer` import

# %%
gpt_model_name = 'gpt2'
bert_model_name = "bhadresh-savani/bert-base-uncased-emotion"

rnd_bert = AutoModelForSequenceClassification.from_config(
    config=BertConfig(
        output_attentions = True
    )
    )
rnd_bert.eval()

bert = AutoModelForSequenceClassification.from_pretrained(
    bert_model_name,
    output_attentions = True
)
bert.eval()

gpt2 = AutoModel.from_pretrained(
    gpt_model_name,
    output_attentions = True
)
gpt2.eval()

# %%
tokenizer_bert = AutoTokenizer.from_pretrained(bert_model_name)
tokenizer_gpt = AutoTokenizer.from_pretrained(gpt_model_name)

# %% [markdown]
# * Encode the sentence using both tokenizers
#     * Return the sentences as torch tensors
# * Save the tokenized sequence as a list of strings in a variable (required for plotting below)

# %%
sentence = "I was so relieved after the phone call yesterday. I smiled the whole day."
enc_bert_sentence = tokenizer_bert(sentence)
enc_gpt_sentence = tokenizer_gpt(sentence)

tokens_bert = tokenizer_bert.convert_ids_to_tokens(
    enc_bert_sentence['input_ids']
)

tokens_gpt = tokenizer_gpt.convert_ids_to_tokens(
    enc_gpt_sentence['input_ids']
)

# %% [markdown]
# * Run the sequence through all 3 models
#     * Extract and save the attention outputs
#     * Disable gradient calculation to save memory and speed up the process

# %%
ids_bert_sentence = torch.tensor(enc_bert_sentence['input_ids']).view(1,18)
ids_gpt_sentence = torch.tensor(enc_gpt_sentence['input_ids']).view(1,16)

with torch.no_grad():
    output_bert = bert(
        input_ids=ids_bert_sentence,
        output_attentions=True
    )
    attention_bert = output_bert.attentions

    output_rnd_bert = rnd_bert(
        input_ids=ids_bert_sentence,
        output_attentions=True
    )
    attention_rnd_bert = output_rnd_bert.attentions

    output_gpt2 = gpt2(
        input_ids=ids_gpt_sentence,
        output_attentions=True
    )
    attention_gpt = output_gpt2.attentions


# %% [markdown]
# * Visualize the attention maps using `bertviz`'s [model_view](https://github.com/jessevig/bertviz#model-view) and [head_view](https://github.com/jessevig/bertviz#head-view) for each of the three models
# * Inspect the attention patterns and structures and answer the questions below
# * If you encounter a `javascript: require is not defined` error, simply execute the cell again, sometimes jupyterlab's widgets need to be reloaded

# %%
model_view(attention_bert, tokens_bert)

# %%
model_view(attention_rnd_bert, tokens_bert)

# %%
model_view(attention_gpt, tokens_gpt)

# %%
head_view(attention_rnd_bert, tokens_bert)

# %%
head_view(attention_bert, tokens_bert)

# %%
head_view(attention_gpt, tokens_gpt)

# %% [markdown]
# Hint: State approximately 2-3 observations/answers per question. Always point to the specific layer(s) and head(s) where your observation(s) can be found.
# 
# Questions:
# * What difference can you observe between the randomly initialized model and both trained models?
# * What general patterns and differences in attention maps can you see between the trained BERT and GPT-2 models? How can they be explained? Think about the attention and masking behavior of BERT vs. GPT models.
# * Compare the trained BERT and GPT-2 model on the following aspects:
#     * Hierarchical patterns: Do lower or higher layers attend to more local and syntactic structures?
#     * Intra-layer head behavior: Do heads belonging to the same layer capture similar or different structures or patterns?
#     * Redundancy: Are there repeated attention patterns across heads and/or layers? Is redundancy rather beneficial or hurtful?
#     * Interpretability: Can you find interpretable structures in certain layers and/or heads? You could try to look for DET-NOUN dependencies, subject-verb-object structures, next- or previous-word attention. For instance, GPT-2's layer 5 head 4 attends 'smiled' to 'the whole day'.
# 

# %% [markdown]
# ___
# Student answers here:
# 
# Q1. In the layers of randomly initialized model, there is no specific pattern for each head. Instead, each head has almost identical attention matrices (values in the matrices are almost uniformly distributed). Given that this is randomly initialized model, there is not any tendence among tokens in attention matrices. Yet, in trained models, there are patterns in each layer's heads. That being said, each layer seems to catch different contextual part of the sentence. 
# 
# Q2. In BERT, attention maps tend to give values more sparsely. This is because, BERT is bidirectional and tokens can see all the sentence. It is sparse because it employs masking behavior, so that the focus is spread across various tokens. However, a token in GPT model can give attention to the ones already appreared in the sequence. The arrows from tokens (left) to the top right token ([CLS]) is more obvious. Since GPT does not use any masking, the patterns are more gradual, accumulated to the previously seen tokens. 
# 
# Q3. BERT vs GPT on the following aspects:
# 
# - Hiearchical patterns: Similar. They both capture more local structures (neighborhoods) in lower layers, whereas in middle-hayer layers the focus becomes more specific and to the point (general), the attention is higher on particular tokens. The only difference is, the last layer of BERT focuses again on the general structure (sparse distribution), this is not observed for GPT.
# - Intra-layer head behavior: BERT has diverse heads within layers. Each head of one layer focuses on different contextual information. However, GPT seems less diverse. More similar patterns can be observed for heads in each layer.
# - Reduncany: BERT has less similar heads, less redundancy. However, it exists some redundant heads across layers. GPT has more redundant attention maps since it is unidirectional. Redundancy is mostly beneficial since it provides more robust attention maps.
# - Interpretability: for GPT, layer 8 head 7 attends 'after' to 'the phone call'. For BERT, layer 5 head 1 attends 'relieved' to 'after the'. There are definitely some interpretable structures.
# ___

# %% [markdown]
# ## Task 1.2: Role and Importance of Individual Heads

# %% [markdown]
# ### Task 1.2.1: Entropy Per Head

# %% [markdown]
# * We saw in our visualizations that sometimes a single head attended only to one specific token, but also that often an individual head attended almost uniformly to all tokens.
# * This phenomenon can be measured, i.e. quantified, using [Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)
# * First, implement the following entropy formula:
# $$
# H(X) := - \sum_{x \in X} p(x) \log{p(x)}
# $$
# * Our attention output tensor of a single layer $L$ of shape `[batch_size, num_heads, seq_len, seq_len]` represents the discrete random variable $X$
# * $X$ consists of `num_heads=12` attention heads in the case of BERT
# * Each attention head has a sequence length of `seq_len` (whatever you chose above for your sentence)
# * $p(x)$ represents the probability distribution over the `seq_len` tokens in the sequence
# * Each $x \in X$ corresponds to a token position in the sequence
# * For each $x \in X$, the value is between 0 and 1, indicating the probability of the attention weight for the token at position $x$ connecting to a subset of tokens within the sequence
#     * Remember that due to the softmax operation during attention calculation, each attention head of our output already represents a probability distribution over the token sequence

# %%
def entropy(p):
    log_p = torch.log(p + 1e-9)
    return -1 * torch.sum(p * log_p, dim=-1)

# %% [markdown]
# * Apply the function to all attention layers
#     * Per layer, the resulting entropy output should be of shape `[batch_size, num_heads, num_tokens]`
# * Calculate the entropy per head by summing up the per-token entropy
#     * Repeat the process for all layers in the model
# * Compare the entropy across heads and layers by visualizing the trends in a line plot
#     * The x-axis should represents the 12 head positions
#     * The y-axis should represent the entropy per head
#     * The plot should include 12 lines (i.e. the layers) with 12 positions (i.e. the heads)
#     * Create an interactive plot by using [plotly.express.line](https://plotly.com/python-api-reference/generated/plotly.express.line)
#         * Complete the function `create_plotly_lineplot`
#         * As the documentation shows, it expects a DataFrame `df` where each column is a layer, and each row are the entropy values of an attention head in that layer
#         * The `plotly.express.line` function returns a [Figure](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html)
#         * We can [fig.update_layout](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.update_layout) to include [custom buttons](https://plotly.com/python/custom-buttons/) to toggle between the lines
#         * To achieve this, we need to create one dictionary per layer that has a `visible` setting to `True` at its layer number, i.e. if its the third layer, the third index should be `True`, and all remaining set to `False`.
#         * In the end, we need a list with 12 dictionaries in the following style
#         ```
#         {
#             "label": "Layer 0",           # the name of the button
#             "method": "update",           # means that we should 'update' the plot when clicking this
#             "args": [                     # the visibility argument for the respective layer
#                 {                         
#                     "visible": [
#                         True,             # set True for the corresponding index of the layer
#                         False,
#                         False,
#                         False,
#                         False,
#                         False,
#                         False,
#                         False,
#                         False,
#                         False,
#                         False,
#                         False,
#                     ]
#                 }
#             ],
#         }
#         ```
#         * Add to the list a `Show All` button with visibility settings to all `True`
#         * Then, use `fig.update_layout`'s `updatemenus` (as linked above) to include the buttons in the plot
#         * Lastly, use [fig.update_traces](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.update_traces) with `mode="lines+markers"` to include markers at each x-tick in the plot
#         * Return the figure at this stage
#     * Save the plot [as an html-file](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.write_html). You can now open it in your browser and analyze the entropy per layer and head. You can also hover over each line and see the exact entropy value for that layer and head.
#     * You can do this interactive plotting style for [all kinds of plots](https://plotly.com/python-api-reference/generated/plotly.express.html#module-plotly.express), which makes analyzing high-dimensional data much more intuitive
#     * Include a screenshot of the html view in your submission
# * Go back to the visualized BERT attention maps and check that some of the patterns here match with the observed visualizations
#     * Choose some distinct results (e.g. outliers) and compare it with the `head_view`
#     * In the drop-down menu, select the outlier layer
#         * (1) Hover of the words to see the color patterns of each respective head
#         * (2) Double click on the head to see only the attention weights and connections of that head
# * Repeat for some (3 or more) patterns and describe what you see
#     * What do high entropy heads correspond to when visualized?
#     * What do low entropy heads correspond to when visualized?

# %%
def create_plotly_lineplot(df, title):
    n_layers = df.shape[1]
    n_heads = df.shape[0]

    fig = px.line(
        df,
        title= title,
        labels={'index':'Heads', 'value':'Entropy', 'variable':'Layers'}
    )

    buttons = [
        {
            'label': f"Layer {i}",
            'method': 'update',
            'args': [
                {
                    'visible': [True if i == j else False for j in range(n_layers)]
                }
            ]
        } for i in range(n_layers)
            ]

    buttons.append(
        {
            'label': 'Show all',
            'method': 'update',
            'args': [
                {
                    'visible': [True for i in range(n_layers)]
                }
            ]
        }
    )

    fig.update_layout(
        updatemenus=[
            {
                'buttons':buttons,
                'direction':'down',
                'showactive':True,
                'x':0.5,
                'xanchor':'left',
                'y':1.0,
                'yanchor':'top'
            }
        ]
    )

    fig.update_traces(mode='lines+markers')

    return fig

# %%
entropy_bert = [entropy(layer) for layer in attention_bert]
entropy_gpt = [entropy(layer) for layer in attention_gpt]
per_head_bert = [torch.sum(layer, dim=2) for layer in entropy_bert]
per_head_gpt = [torch.sum(layer, dim=2) for layer in entropy_gpt]
# 12 heads for 12 layers each

# %%
dict_of_layers = {i: per_head_bert[i].numpy() for i in range(len(per_head_bert))}
df = pd.DataFrame({key: value.flatten() for key, value in dict_of_layers.items()})

# %%
create_plotly_lineplot(df, title="Layers' Entropy trends across heads").write_html('dl4nlp_ex04_task1.html')

# %%
# for visibility
create_plotly_lineplot(df, title="Layers' Entropy trends across heads")

# %% [markdown]
# ___
# Student answers here:
# 
# - Low entropy:
#     - Example 1: Layer 1 Head 6 attends "day" to "the" (also "call" to "the" too). DET-NOUN relationship catched.
#     - Example 2: Layer 7 Head 3 attends "call" to "phone (noun-noun), "so" to "i was" ("so" as adverb modifiers) and "whole" to "the" and "day" (det-noun)
#     - Regarding to these examples, If the entropy of the attention map is low, the model can effectively capture specific contexts.
# - High entropy:
#     - Example 3: Layer 0 Head 4 couldn't capture any context given the fact that the distribution of attention is more evenly among tokens. 
# ___

# %% [markdown]
# ### Task 1.2.2: Importance Per Head

# %% [markdown]
# * After having seen how different attention heads correspond to different entropy levels, we can also calculate an [importance score](http://arxiv.org/abs/1905.10650) for each head
# * For this experiment, we define a mask $\xi_{h}$, which can drop out (i.e. set to zero) an attention head
#     * If the mask is not active (i.e. set to 1), the attention weights of the head remain unchanged (represented by $\xi_h$)
# * Then, we feed in a sample $x$, calculate the loss $\mathcal{L}(x)$, and analyze the sensitivity of the loss w.r.t. the masked head
# * The importance of that head follows as
# $$
# I_h = \big\vert \frac{\partial \mathcal{L}(x)}{\partial \xi_h} \big\vert
# $$
# * The absolute value avoids large negative and positive values from nullifying each other

# %% [markdown]
# * Define the `device` to use the GPU

# %%
device = 'cuda:0'

# %% [markdown]
# * For this experiment, we will work with the [DAIR-AI Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion), but in this case you only need to load the `split='test'`
#     * Select only the first 32 samples
#     * Tokenize, pad, truncate to maximum 512 tokens, and return torch tensors of those 32 samples
#     * Extract the labels of those 32 samples, too
#     * This batch will serve as the exemplary classification dataset on which we calculate importance metrics
# 

# %%
emotion_dataset = load_dataset('dair-ai/emotion', split='test')
sampled_dataset = emotion_dataset.select(range(32))
tokenized_data = tokenizer_bert(
    sampled_dataset['text'], 
    truncation=True, 
    padding=True, 
    max_length=512,
    return_tensors='pt'
    )
labels = torch.tensor(sampled_dataset['label'], device=device)

# %% [markdown]
# * Then, we will write a function `get_head_importance`, which takes in the `model`, `samples`, `labels`, `device`, and an optional `mask`
# * Define a zero-tensor of shape `[num_layers, num_heads]`, which stores the importance scores of all heads
# * If no `mask` is provided, create a `mask` of ones (i.e. we are not masking anything currently) in the same shape
# * In all cases, whether given or newly created, [enable gradient calculuation](https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad_.html) for the mask
# * Move all necessary objects to the `device`
# * Forward the samples through the model
#     * Include the [labels](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification.forward.labels) and [head_mask](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification.forward.head_mask)
#     * Extract loss and logits from the [output](https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput)
# * Then, call `backward()` on the loss to distribute the gradients among the head mask
#     * We do not train the model (i.e. no optimizer here), but we need the gradients (see formula above) for each head mask position w.r.t. each loss sample
#     * PyTorch's autograd and `backward()` method allows us calculate and backpropagate any form of gradients, even without the classical training/optimization setup
# * Now, we can check the head importance by accessing the [grad](https://pytorch.org/docs/stable/generated/torch.Tensor.grad.html) method of the earlier created `mask` tensor
#     * [detach](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html) the tensor from the gradient calculation graph
#     * Take the absolute value of it as shown in the formula
# * It is common to normalize the head-importance scores in two ways
#     * First, divide the head importance scores by the number of real (i.e. non-padding) tokens in the batch
#     * Secondly, calculate the [l2-norm](https://mathworld.wolfram.com/L2-Norm.html) per layer
#         * Make sure to adjust the shapes of the $l_2$ norm to fit the head importance tensor
#         * Add a safety term of `1e-20` to the $l_2$ norm to avoid possible divisons by zero
#         * Divide the head importance by the $l_2$ norm
# * Return the head importance, logits, and labels

# %%
def get_head_importance(model, samples, labels, device, mask=None):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_real_tokens = sum(sum(row) for row in samples['attention_mask'])
    #samples = samples.to(device)
    # debugging
    samples = {key: value.to(device) for key, value in samples.items()}
    labels = labels.to(device)
    model = model.to(device)

    imp_scores = torch.zeros([n_layers, n_heads], device=device)
    if mask==None:
        mask = torch.ones([n_layers, n_heads], device=device)
    else:
        mask.to(device)
        
    mask.requires_grad_(True)

    output = model(
        **samples,
        labels=labels,
        head_mask=mask
    )
    
    loss = output.loss
    logits = output.logits
        
    loss.backward()
    imp_scores += mask.grad.detach().abs()
    imp_scores_normalized = imp_scores / n_real_tokens

    l2_norms = torch.norm(imp_scores_normalized, p=2, dim=1).unsqueeze(1).expand(-1, n_heads) + 1e-20
    imp_scores_normalized /= l2_norms
    
    return imp_scores_normalized, logits, labels

# %% [markdown]
# * Calculate the head importance for our BERT model and the 32 samples

# %%
head_imp_bert = get_head_importance(bert, tokenized_data, labels, device)

# %% [markdown]
# * Visualize the head importance in the same interactive way as you did with the entropy plots
# * You can re-use `create_plotly_lineplot`, just create a DataFrame from the returned head importance tensor instead of the entropy tensor
# * Comment on any trends you see in the importance of heads. Which head is the most important?
# * Create again an HTML version of the plot and include a screenshot of your plot in the submission

# %%
imp_scores_normalized, _, _ = head_imp_bert
imp_scores = imp_scores_normalized.detach().cpu().numpy()

dict_of_scores = {i: imp_scores[i] for i in range(len(imp_scores))}
df_imp_score = pd.DataFrame({key: value.flatten() for key, value in dict_of_scores.items()})

create_plotly_lineplot(df_imp_score, title="Head Importance").write_html('dl4nlp_ex04_task1_2.html')
create_plotly_lineplot(df_imp_score, title="Head Importance")

# %% [markdown]
# ___
# Student answers here: Layer 7 Head 0 is the most important one. Among the heads, 0 and 4th heads seem better than the overall. After head 7, none of the heads within layers could achieve more than 0.5 importance. Given that, I would conclude that the earlier heads are more important and after some point heads tend to contribute relatively lower. Other than that, there is no consistent winner among heads and layers. Some heads are important for particular layers, some not. 
# 
# ___

# %% [markdown]
# * Investigate whether there is a positive, negative, or no correlation between each attention head's entropy and importance score
# * Calculate the [correlation coefficient](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html) between the entropies and importance scores
# * Create a scatter plot to visualize your findings
# * Comment on the results

# %%
for i in range(df.shape[0]):  # for each head
    head_entropies = df.iloc[i].values
    head_importances = df_imp_score.iloc[i].values
    head_corr = np.corrcoef(head_entropies, head_importances)[0, 1]
    print(f"Head {i} correlation: {head_corr}")

# %%
plt.style.use("seaborn-v0_8-darkgrid")

plt.figure(figsize=(6, 6))

plt.scatter(df, df_imp_score)
plt.xlabel('Entropy')
plt.ylabel('Importance Score')
plt.title('Correlation between Entropy and Importance Score among heads')
plt.grid(True)
plt.show()

# %% [markdown]
# ___
# Student answers here: The correlation between entropy and importance score is weak but positive. Heads with lower entropy are not consistently more or less important than the others. Briefly, I would conclude that the importance score vary independently of how focused each attention to a specific token.
# 
# ___

# %% [markdown]
# ### Task 1.2.3: Masking and Pruning Heads

# %% [markdown]
# * As we can see from the line and scatter plots, a lot of attention heads have an importance of near 0
# * As a consequence, we can investigate which and how many we can remove from our model without losing a predefined threshold of performance
# * Dropping attention heads leads to dropping parameters, which results in a smaller model for further finetuning or inference
# * However, directly dropping heads is risky, since dropping a head affects the calculations and, therefore, the performance of the remaining model
# * Therefore, we remove heads through masking consecutively and test the performance after each removed head
# * Once we found an acceptable performance-masking tradeoff, we can actually remove the heads from the model

# %% [markdown]
# * Write a function that takes in a `model`, `samples`, `labels`, `threshold` and `device`
# * Start again with an inactive mask
# * Set the starting head importance, logits, and labels
# * Use logits and labels to calculate a base accuracy performance with the inactive mask
# * We perform the following process until either the accuracy dropped more than 5% (i.e. threshold level) below starting accuracy, or until all heads are masked
#     * We continuously select the next lowest importance head
#     * We mask its position in the mask
#     * Recalculate the head importance and accuracy level with the masked heads
#       * Hint: Depending on your implementation, you might find [clone](https://pytorch.org/docs/stable/generated/torch.clone.html) and/or [detach](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html) helpful when dealing with the `mask` and/or `head_importance`
#     * Set the head's importance to positive infinity so that it will always be last in future importance rankings
# * Save the number of masked heads with the corresponding accuracy level for each iteration as a tuple `(int, float)` in a list
#     * The accuracy should be rounded to 4 decimal places
# * The function returns the final `mask` and the list of tuples

# %%
def find_min_heads(model, samples, labels, threshold, device):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    mask = torch.ones([n_layers, n_heads], device=device, requires_grad=True)
    head_imp = torch.zeros([n_layers, n_heads], device=device)

    output = model(
        **samples,
        labels=labels,
        head_mask=mask
    )

    loss, logits = output.loss, output.logits
    loss.backward()
    head_imp += mask.grad.detach().abs()

    num_correct = int(torch.sum(torch.argmax(logits, dim=1) == labels))
    st_accuracy = num_correct / len(labels)
    acc = st_accuracy

    log_list = list()
    masked_heads = 0

    while (abs(st_accuracy - acc) < threshold) & (mask.sum().item() > 0):
        masked_heads += 1
        ind = (head_imp == head_imp.min()).detach()

        mask = mask.detach()
        mask[ind] = 0
        head_imp[ind] = torch.inf
        mask = mask.requires_grad_(True)

        output = model(
            **samples,
            labels=labels,
            head_mask=mask
        )

        loss, logits = output.loss, output.logits
        loss.backward()

        head_imp += mask.grad.detach().abs()
        num_correct = int(torch.sum(torch.argmax(logits, dim=1) == labels))
        acc = num_correct / len(labels)

        log_list.append((masked_heads, round(acc, 4)))       
        
    return mask, log_list

# %% [markdown]
# * Find the minimum number of heads required before performance drops below the given 5% threshold
# * Plot the accuracy level per number of masked heads below
# * Comment on your findings
#     * Keep in mind that we are evaluating on a very small set of 32 samples
#         * I.e extreme results are possible, such as masking large percentages of the model's heads without noticing performance differences
#     * It is possible that the accuracy can rise intermediately, even though heads are dropped. Try to explain what is happening in such cases.
#     * In practice, you would usually run this experiment on the whole [GLUE dataset](https://huggingface.co/datasets/glue) to have a more representative and diverse accuracy baseline

# %%
mask, logs = find_min_heads(
    bert.to(device),
    tokenized_data.to(device),
    labels.to(device),
    0.05,
    device
)

masked_heads = [i[0] for i in logs]
accuracy_logs = [i[1] for i in logs]

fig = plt.figure(figsize=(6,6), facecolor='white')
plt.plot(masked_heads, accuracy_logs)
plt.title('Accuracy change with masked heads')
plt.xlabel('# of masked heads')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

# %% [markdown]
# ___
# Student answers here: We can keep our accuracy at 100% even after dropping a bunch of heads. Given that our sample size is quite small, minor shifts in predictions can result in remarkable changes in accuracy. That's why, it is normal to see increasing accuracy at certain points. Many heads have a little influence on the predictions, so that their removal causes little or no downgrade in performance.
# ___

# %% [markdown]
# * Now that we have our mask, we can actually remove from the model the heads we masked before (pruning)
# * Create a dictionary that [maps each layer's index to the indices of the heads to prune](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.prune_heads)
# * Calculate the number of parameters before and after pruning to show the effect
#     * Note that the `prune_heads` method changes the model in-place

# %%
params_before = sum(p.numel() for p in bert.parameters())
prune_heads = {i: [j for j in range(12) if mask[i][j] == 0] for i in range(12)}
bert.prune_heads(heads_to_prune=prune_heads)
params_after = sum(p.numel() for p in bert.parameters())

print(f'# of Params before: {params_before}')
print(f'# of Params after: {params_after}')

# %%
params_after/ params_before - 1

# %% [markdown]
# # Task 2: Decoding

# %% [markdown]
# * In the following, we will have a look at various ways of generating text from model output probabilities
# * Specifically, we will see how much of an impact the decoding strategy has on the generated text
# * Using Hugging Face, we will try both basic and more advanced as well as commonly used decoding strategies for the same input prompt
# * Therefore, we can see and analyze the shortcomings and advantages of various decoding strategies *while keeping the model and prompt equal*

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# %% [markdown]
# * Define the device for decoding
# * Load GPT-2 with its language modeling head and the corresponding tokenizer
#     * Since we are doing open ended text generation, there is no padding token necessary
#     * To disable the warning by the model, you can set `pad_token_id=tokenizer.eos_token_id` inside the `from_pretrained` model loading process
#     * Set it into eval mode

# %%
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
model.eval()

# %% [markdown]
# ## Task 2.1: Greedy Search

# %% [markdown]
# * Greedy search is the most straightforward method for generating sequences
# * Starting from an initial context, e.g. a prompt, the model generates the next token in the sequence by selecting the token that has the highest predicted probability according to the model
# * The selected token is added to the sequence, and the process is repeated to generate the entire sequence
# * Importantly, in each step the model *only* considers the most likely token based on its current context
# * Greedy search is deterministic, meaning it always chooses the same token given the current context
# * The decoding algorithm continues until the set maximum generated length has been reached, or until the end-token has been generated

# %% [markdown]
# * Given the sentence below:
#     * Tokenize the prompt
#     * Generate the output using [model.generate()](https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)
#         * Limit the generation length to 128 tokens
#     * Decode the generated output to a human readable format
#     * Discuss the quality of output. Relate the possible advantages and disadvantages of the output to the algorithm used.

# %%
sentence = "I came back from holidays and"
tokenized_sentence = tokenizer(sentence, return_tensors='pt')
output = model.generate(**tokenized_sentence, max_length=128)
decoded_output = tokenizer.decode(output[0])
decoded_output

# %% [markdown]
# ___
# Student answers here: 
# For the underlying example, model can continue the sentence with a correct approach. The advantage is that model can generate contextually and grammatically correct, meaningful sentence. However, the output is repetitive and an adjustment has to be made in order to prevent this.
# ___

# %% [markdown]
# ## Task 2.2: Beam search

# %% [markdown]
# * Beam search is an extension of greedy search and aims to overcome some of its limitations
# * Starting from an initial context, the model generates multiple candidate tokens for the next position, typically referred to as "beams"
# * Each beam represents a hypothesis for the next token in the sequence
# * The model scores and ranks these beams based on their predicted probabilities
# * The top-k beams with the highest scores are selected to continue the generation process, where k is a user-defined parameter known as the "beam width"
# * The selected beams are extended with candidate tokens, and the process is repeated iteratively
# * Beam search maintains multiple active hypotheses in parallel, allowing it to optimize coherence in the sequence across multiple paths
# * Decoding continues until one or more beams generate an end token or reach the maximum sequence length

# %% [markdown]
# * In the following, we will implement (a simplified version of) the beam search algorithm, and compare it to Hugging Face's implementation of beam search
#     * Hint: Aside from efficiency tweaks, we do not include any length bonus or penalization of short sequences, as you may later notice (depending on the prompt)
# * First, we define a helper function called `generate_candidates`, which will generate our possible next token along each hypothesis path
#     * It takes as input the `model`, the `context`, which is the tokenized prompt, the so far newly generated `sequence`, the `beam_width`, and the `device`
#     * The model predicts logits based on the concatenated context and sequence
#     * We extract the last predicted token position of the logits, which serves as the next token prediction
#     * From this hidden representation, we return the top-k indices of the vocabulary, whereas the `beam_width` parameter determines the $k$
#     * Make sure to place everything on the `device` as required
# * The main `beam_search` function takes as input the `model`, `context`, `beam_width`, `max_length`, and `device`
#     * Create a data structure `hypotheses` that holds all of the newly generated sequences along with their log-probability scores
#     * In our simplified version, we generate new tokens until the maximum length is reached
#     * In each step, we iterate through all hypotheses
#     * Each hypothesis generates candidates with our above helper function based on the current progress of newly generated sequences
#         * In the first iteration, this newly generated sequence is simply empty (it is not necessary to consider start tokens here)
#         * As a result, we generate the first new token only based on the provided `context`
#     * Next, we extend the hypothesis step by step with each candidate token
#         * Each extension is paired with the initial context and put through the model
#         * Then, we calculate log-probabilities for each position in the vocabulary and extract the candidate position
#         * This log-probability is stored together with the hypthesis and extended/summed up with each new candidate token and log-probability
#         * Save all scored hypothesis and candidate combinations
#     * Based on all hypothesis combinations, extract the top-`beam_width` number of combinations based on the highest scores
#     * Overwrite the earlier created `hypotheses` datastructure with those top-`beam_width` hypotheses, so that we start the next iteration with `beam_width`-many hypotheses
#     * When `max_length` token have been generated, return the combined indices of the initial context and the generated ones
#     * Disable gradient calculation and move everything to `device` as necessary
# * Use the function with `beam_width=5` and `max_length=50` to generate the output, then decode it as done above.

# %%
def generate_candidates(model, context, sequence, beam_width, device):
    input = torch.cat([context.to(device, dtype=torch.long), sequence.to(device, dtype=torch.long)], dim=-1)
    model = model.to(device)
    input = input.to(device)
    
    with torch.no_grad():
        output = model(input_ids = input)
        logits = output.logits

    next_token_logits = logits[:, -1, :]
    top_k_scores, top_k_indices = torch.topk(next_token_logits, beam_width, dim=-1)

    return top_k_indices.squeeze(0), top_k_scores.squeeze(0)

# %%
def beam_search(model, context, beam_width, max_length, device):
    hypotheses = [{'seq': torch.zeros(1, 0, device=device), 
                   'log_score':0.0
                   }
                   ]
    counter = 0

    while counter <= max_length:
        hyp_ext = list()
        counter += 1
        for hyp in hypotheses:
            sequence = hyp['seq']
            top_indices, scores = generate_candidates(model, context, sequence, beam_width, device)
            
            # Extend each hypothesis with the top-k tokens
            for token, log_prob in zip(top_indices, scores):
                token = token.unsqueeze(0).unsqueeze(0).to(device)
                new_input = torch.cat([sequence, token], dim=-1)
                new_score = hyp['log_score'] + log_prob.item()

                hyp_ext.append({'seq': new_input, 'log_score': new_score})
        
        hypotheses = sorted(hyp_ext, key=lambda x: x['log_score'], reverse=True)[:beam_width]
    
    return hypotheses

# %%
hyp_test = beam_search(model, tokenized_sentence['input_ids'], beam_width=5, max_length=50, device=device)

# %%
best_seq = max(hyp_test, key=lambda x: x['log_score'])['seq'][0]
decoded_text = tokenizer.decode(best_seq.long().tolist(), skip_special_tokens=True)
print(sentence + '' + decoded_text)

# %% [markdown]
# * Now repeat the beam search process using [model.generate()](https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)
#     * Limit the generation length to 128 tokens
#     * To choose beam search, set `num_beams=5` and `early_stopping=True`
# * Decode the generated output to a human readable format
# * Discuss the quality of our simplified beam search output and Hugging Face's output. Relate the possible advantages and disadvantages of the output to the algorithm used.

# %%
tokenized_sentence = tokenized_sentence.to(device)
model = model.to(device)

beam_output = model.generate(
    **tokenized_sentence,
    max_new_tokens=128,
    num_beams=5,
    early_stopping=True
    )

print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# %% [markdown]
# ___
# Student answers here: Our function is more flexible since it is not pre-built. We can restrict some combinations (based on tokens), we are also be able adjust different part of the method. However it is potentially less efficient due to naive operations compared to HF's function. There is no early stopping in our function, which is also a downside.
# 
# Overall Beam Search Output Advantages: consider multiple candidates instead of a greedy one choice, more coherent sentences
# Disadvantages: computationally expensive due to broader search
# ___

# %% [markdown]
# * In order to specifically reduce repetitions, beam search can be adapted with a `no_repeat_ngram_size=2` option
#     * This prevents the model from generating any ngrams of size 2 twice
# * Generate its output using the option and discuss the new results again. When can this option be useful? In which case does it (always) deteriorate the output?

# %%
beam_output = model.generate(
    **tokenized_sentence,
    max_new_tokens=128,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=2
    )

print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# %%
beam_output = model.generate(
    **tokenized_sentence,
    max_new_tokens=128,
    num_beams=50,
    early_stopping=True,
    no_repeat_ngram_size=2
    )

print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# %% [markdown]
# ___
# Student answers here:
# The new result is more natural since it prevents redundancy. It might be deteriorative, in case repetition is necessary or the vocabulary is too small. Also, in case num_beams is high (like 50), the result becomes unnatural with no_repeat_ngram_size=2.
# ___

# %% [markdown]
# ## Task 2.3: Sampling
# * Sampling decoding is probabilistic approach for generating sequences of tokens
# * Starting from an initial context, the model generates the next token in the sequence by randomly sampling from the distribution of predicted token probabilities
# * Instead of deterministically selecting the most likely token as in greedy search, sampling decoding introduces randomness by considering the predicted probabilities as a probability distribution
# * The model assigns probabilities to each possible token, and the next token is chosen probabilistically based on these probabilities
# * Tokens with higher probabilities are more likely to be selected, but there is still an element of randomness involved
# * Sampling decoding can be guided by a temperature parameter, where higher temperatures increases the likelihood of highly probable words, and lower temperatures smoothes out the distribution
# * The decoding continues iteratively, with each token influencing the distribution of probabilities for the next token.
# * The process is repeated until an end token is generated, or the maximum sequence length is reached

# %% [markdown]
# * Generate the output using [model.generate()](https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)
#     * Limit the generation length to 128 tokens
#     * To choose sampling decoding, activate `do_sample=True`, set `top_k=0` and set the temperature e.g. to `temperature=0.6`
# * Decode the generated output to a human readable format
# * Discuss the quality of output. Relate the possible advantages and disadvantages of the output to the algorithm used.

# %%
sampling_output = model.generate(
    **tokenized_sentence,
    max_new_tokens=128,
    do_sample=True,
    top_k=0,
    temperature=0.6
    )

print(tokenizer.decode(sampling_output[0], skip_special_tokens=True))

# %% [markdown]
# ___
# Student answers here:
# Quality of the output is relatively lower. There is lack of coherence and grammatically correctness.
# Advantages: More diverse and creative text generation, introducing noise might be a good idea to make it more human-like text
# Disadvantages: lack of coherence, it is apparently sensitive to hyperparameters, due to its probabilistic nature there might be non-sense outputs
# ___

# %% [markdown]
# ## Task 2.4: Top-K Sampling

# %% [markdown]
# * Top-k sampling amends the above sampling approach by considering only a restricted set of the most likely tokens per step
# * Instead of sampling from the entire vocabulary of possible tokens, top-k sampling limits the selection to the top-k tokens with the highest predicted probabilities
# * K is a user-defined parameter and determines the size of the set of candidates
# * The model assigns probabilities to each possible token, ranks them based on their predicted probabilities, and selects from the top-k tokens for the next position in the sequence
# * This approach combines randomness within a the restricted set of candidates
# * Tokens within the top-k can still be selected probabilistically, with tokens having higher probabilities being more likely to be chosen
# * The decoding continues iteratively until and end token is generated, or the maximum sequence length is reached

# %% [markdown]
# * Generate the output using [model.generate()](https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)
#     * Limit the generation length to 128 tokens
#     * To choose top-k sampling decoding, activate `do_sample=True` and set `top_k=50`
# * Decode the generated output to a human readable format
# * Discuss the quality of output. Relate the possible advantages and disadvantages of the output to the algorithm used.

# %%
top_k_output = model.generate(
    **tokenized_sentence,
    max_new_tokens=128,
    do_sample=True,
    top_k=50,
    #temperature=0.6
    )

print(tokenizer.decode(top_k_output[0], skip_special_tokens=True))

# %% [markdown]
# ___
# Student answers here:
# Quality is increased compared to sampling decoding method. The sentences are more coherent and logical.
# Advantages: creativity is kept and controlled, avoiding non-sense sentences with restricting token sample
# Disadvantages: still sensitive to hyperparameters, avoiding some tokens restrict creativity as well
# ___

# %% [markdown]
# ## Bonus: Contrastive Search (bonus)

# %% [markdown]
# * [Contrastive search](https://arxiv.org/abs/2202.06417) addresses text generation issues by introducing a learnable decoding framework constisting of two components
# * Contrastive Training: **Sim**ple **c**ontrastive framework for neural **t**ext **g**eneration (SimCTG)
#     * Aims to improve the quality of token representations generated by language models
#     * It trains the model to learn discriminative token representations
#     * This greatly assists the model to produce more coherent and contextually relevant text
# * Contrastive Search:
#     * Complements CTG by first calculating a confidence score among the top-k candidate tokens represented by the model probabilities
#     * Then, a degeneration penalty is introduced
#     * It measures how discriminative the top-k candidate tokens are w.r.t. the previous context
#     * The cosine similarity is used as a measure
#     * The larger the degeneration penalty score is, the more similar the next token is to the previous context, and the more likely it is that the output degenerates
# * A hyperparameter $\alpha$ regulates the tradeoff between the model confidence and the degeneration penalty
#     * If $\alpha=0$, greedy search is performed

# %% [markdown]
# * In terms of performance, contrastive search was a big step up over previous algorithms
# * However, the algorithm was only recently implemented in the Transformers library (i.e. not available in the 4.19.4 version we used before)
# * If you want to try it, install versions >=4.33
#     * Depending on the prompt, the sequence could be more relevant and coherent, as well as less repetitive
#     * However, due to our very small model size, the answers might still be very repetitive and less coherent, but that is the model's fault in this case
#     * See [here](https://github.com/huggingface/transformers/issues/19182#demonstration) for some impressive demonstrations with OPT-6.7b and contrastive search vs. OPT-175b and nucleus sampling (another version top-k sampling)
# 
# Bonus: Code is given, simply try it out if you want
# * Generate the output using [model.generate()](https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)
#     * Limit the generation length to 128 tokens
#     * To choose contrastive search, set `penalty_alpha=0.6`, `top_k=4`, and allow `max_new_tokens=128`
# * Decode the generated output to a human readable format
# * Discuss the quality of output. Relate the possible advantages and disadvantages of the output to the algorithm used.
# 
# ```
# contrastive_out = model.generate(**tokenized, penalty_alpha=0.6, top_k=4, max_new_tokens=128)
# contrastive_decoded = tokenizer.decode(contrastive_out[0], skip_special_tokens=True)
# print('Contrastive search result:')
# print(contrastive_decoded)
# ```

# %%



