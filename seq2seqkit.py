from random import random
from transformers import AutoTokenizer, EncoderDecoderModel

def prep(encoder_model, decoder_model):
	tokenizer = AutoTokenizer.from_pretrained(encoder_model)
	model = EncoderDecoderModel.from_encoder_decoder_pretrained(
		encoder_model,
		decoder_model,
		max_length=40,
	)
	return tokenizer, model

def generate(tokenizer, model, input_ids, input_labels, always_append):
	sep_code = tokenizer.encode('')[-1]
	myrows = []

	batch = tokenizer(input_ids, return_tensors="pt", truncation=True, padding="longest")
	generated = model.generate(batch.input_ids,
							   decoder_start_token_id=model.config.decoder.pad_token_id)
	g_dex = 0
	for g in generated:
		glist = g.tolist()
		if sep_code in glist:
			alt_txt = tokenizer.decode(glist[1 : glist.index(sep_code)])
		else:
			# the longest sequence of the batch (no [SEQ] token at end)
			alt_txt = tokenizer.decode(glist[1 : ])
		# rules on whether to add to final
		if always_append or (alt_txt.lower().replace(' ', '') != input_ids[g_dex].lower().replace(' ', '')):
			myrows.append([alt_txt, input_labels[g_dex]])
		g_dex += 1
	return myrows

def apply_sequenced(pattern, encoder_model, decoder_model, trainrows, frequency, always_append):
	if pattern == "append":
		finalrows = trainrows.copy()
	else:
		finalrows = []
	input_ids = []
	input_txts = []
	input_labels = []

	tokenizer, model = prep(encoder_model, decoder_model)
	for row in trainrows:
		rowtxt = row[0]
		rowlabel = row[1]

		if random() < frequency:
			# avoid sequences being too long
			maxtxt = rowtxt.split(" ")
			if len(maxtxt) > 400:
				rowtxt = " ".join(maxtxt[:400])

			# append to batch
			input_ids.append(rowtxt)
			input_labels.append(rowlabel)

			if len(input_ids) > 150:
				newrows = generate(tokenizer, model, input_ids, input_labels, always_append)
				finalrows += newrows
				input_ids = []
				input_labels = []
		elif pattern == "replace":
			finalrows.append(row)
	if len(input_ids) > 0:
		# leftover from paging
		newrows = generate(tokenizer, model, input_ids, input_labels, always_append)
		finalrows += newrows

	return finalrows

def append_sequenced(encoder_model, decoder_model, trainrows, frequency=0.5, always_append=False):
	return apply_sequenced("append", encoder_model, decoder_model, trainrows, frequency, always_append)

def replace_sequenced(encoder_model, decoder_model, trainrows, frequency=0.5):
	return apply_sequenced("replace", encoder_model, decoder_model, trainrows, frequency, True)
