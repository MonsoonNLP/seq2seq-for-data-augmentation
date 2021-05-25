# seq2seq-for-data-augmentation

Kit of functions to help modify a text dataset, with a seq2seq model

## Features

### Append rows to data

Send the encoder and decoder model names from HuggingFace,
the data as an array ```[[txt1, label1], [txt2, label2]]```,
model's maximum sequence length (default is 512),
frequency of flipping a row (default is 0.5),
and whether to append a row if it came out identical to the original (default is False)


```python
append_sequenced(
    "monsoon-nlp/es-seq2seq-gender-encoder",
    "monsoon-nlp/es-seq2seq-gender-decoder",
    [["Bueno", 0], ["La biblioteca", 1], ["La maestra es tonta", 0]],
    seq_length=512,
    frequency=0.5,
    always_append=False
)
```

If randomly selected, the input ```["La maestra es tonta", 1]``` will result in ```["el maestro es tonto", 1]``` being appended to return data.

If always_append=True, the "bueno" and "la biblioteca" rows
will be included, unmodified.

### Replace rows in data

Send the encoder and decoder model names from HuggingFace,
the data as an array ```[[txt1, label1], [txt2, label2]]```,
model's maximum sequence length (default is 512),
and frequency of flipping a row (default is 0.5)

```python
replace_sequenced(
    "monsoon-nlp/es-seq2seq-gender-encoder",
    "monsoon-nlp/es-seq2seq-gender-decoder",
    [["Bueno", 0], ["La biblioteca", 0], ["La maestra es tonta", 1]],
    seq_length=512
    frequency=0.5
)
```

If randomly selected, the input ```["La maestra es tonta", 1]``` will be replaced with ```["el maestro es tonto", 1]```.

## Applied

Working with SimpleTransformers to see if appending or replacing rows
improves accuracy of classifiers:

https://colab.research.google.com/drive/194ITDA1AjxAx_4ZLjoRFQI1aWzsl7xU8?usp=sharing

## License

Open source, MIT license
