import os
import torch
from tqdm import tqdm, trange
import argparse
import numpy as np
from transformers import xnli_processors as processors
from transformers import xnli_compute_metrics as compute_metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import (
	WEIGHTS_NAME,
	AdamW,
	BertConfig,
	BertForSequenceClassification,
	BertTokenizer,
	DistilBertConfig,
	DistilBertForSequenceClassification,
	DistilBertTokenizer,
	XLMConfig,
	XLMForSequenceClassification,
	XLMTokenizer,
	get_linear_schedule_with_warmup,
	AlbertTokenizer,
	AlbertForSequenceClassification,
	AlbertConfig
)


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_dir",
		default="/home/jqu/Documents/data/XNLI/",
		type=str,
		required=False,
		help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
	)
	parser.add_argument(
		"--model_type",
		type=str,
		required=True,
		help="distilbert|bert"
		)
	parser.add_argument(
		"--model_dir",
		type=str,
		required=True,
		help="where the trained model locates"
		)
	args = parser.parse_args()
	# load test dataset
	processor = processors["xnli"](language="en", train_language="en")
	examples = processor.get_test_examples(args.data_dir)


	if args.model_type == "bert":
		# prepare tokenizer
		tokenizer = BertTokenizer.from_pretrained(
		args.model_dir,
		do_lower_case=False)

		model = BertForSequenceClassification.from_pretrained(args.model_dir)
	
	elif args.model_type == "distilbert":
		tokenizer = DistilBertTokenizer.from_pretrained(
		args.model_dir,
		do_lower_case=False)

		model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)

	elif args.model_type == "albert":
		tokenizer = AlbertTokenizer.from_pretrained(
		args.model_dir,
		do_lower_case=False)

		model = AlbertForSequenceClassification.from_pretrained(args.model_dir)


	model.to("cuda:0")
	model.eval()

	features = convert_examples_to_features(
			examples,
			tokenizer,
			label_list=processor.get_labels(),
			max_length=128,
			output_mode="classification",
			pad_on_left=False,
			pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
			pad_token_segment_id=0,
			mask_padding_with_zero=True)

	# Convert to Tensors and build dataset
	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
	all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

	all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

	dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
	eval_sampler = SequentialSampler(dataset)
	eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=512)

	overall_preds = [[], []]
	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		with torch.no_grad():
			batch = tuple(t.to("cuda:0") for t in batch)
			inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
			if args.model_type != "distilbert":
				inputs["token_type_ids"] = (
					batch[2] if args.model_type in ["bert"] else None
				) # XLM and DistilBERT don't use segment_ids
			outputs = model(**inputs)
			_, logits = outputs[:2]
			preds = logits.detach().cpu().numpy()
			preds = np.argmax(preds, axis=1)
			overall_preds[0] += preds.tolist()

			out_label_ids = inputs["labels"].detach().cpu().numpy()
			overall_preds[1] += out_label_ids.tolist()
	# compute scores
	result = accuracy_score(overall_preds[0], overall_preds[1])
	print(f"Overall accuracy: {result}")
	confusion_score = confusion_matrix(overall_preds[0], overall_preds[1])
	print("confusion matrix:\n")
	print(confusion_score)

if __name__ == "__main__":
	main()
