# seq2seq-for-data-augmentation

Kit of functions to help modify a text dataset with a seq2seq model

## Features

### Append rows to data

Send the encoder and decoder model names from HuggingFace,
the data as a DataFrame ```pd.DataFrame([[txt1, label1], [txt2, label2]], columns=['text', 'label'])```,
model's maximum sequence length (default is 512),
frequency of flipping a row (default is 0.5),
random_state (given to train_test_split),
and whether to append a row if it comes out identical to the original (default is False)


```python
initial_df = pd.DataFrame([["Bueno", 0], ["La biblioteca", 1], ["La maestra es tonta", 0]],
  columns=['text', 'label']
)
append_sequenced(
    "monsoon-nlp/es-seq2seq-gender-encoder",
    "monsoon-nlp/es-seq2seq-gender-decoder",
    initial_df,
    seq_length=512,
    frequency=0.5,
    random_state=1,
    always_append=False
)
```

If randomly selected, the input ```["La maestra es tonta", 1]``` will result in ```["el maestro es tonto", 1]``` being appended to the returned DataFrame.

If always_append=True, the "bueno" and "la biblioteca" rows
will be included, unmodified.

### Replace rows in data

Send the encoder and decoder model names from HuggingFace,
the data as a DataFrame ```pd.DataFrame([[txt1, label1], [txt2, label2]], columns=['text', 'label'])```,
model's maximum sequence length (default is 512),
frequency of flipping a row (default is 0.5),
and random_state (given to train_test_split)

```python
initial_df = pd.DataFrame([["Bueno", 0], ["La biblioteca", 1], ["La maestra es tonta", 0]],
  columns=['text', 'label']
)
replace_sequenced(
    "monsoon-nlp/es-seq2seq-gender-encoder",
    "monsoon-nlp/es-seq2seq-gender-decoder",
    initial_df,
    seq_length=512
    frequency=0.5,
    random_state=12
)
```

If randomly selected, the input ```["La maestra es tonta", 1]``` will be replaced with ```["el maestro es tonto", 1]```.

## Applied

Working with SimpleTransformers to see if appending or replacing rows
improves accuracy of classification or regression tasks:

https://colab.research.google.com/drive/194ITDA1AjxAx_4ZLjoRFQI1aWzsl7xU8?usp=sharing

## Dependencies

```
pip install pandas scikit-learn transformers
```

## License

Open source, MIT license
