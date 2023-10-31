import logging; logger = logging.getLogger(__name__)
import math
import os
import random

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset, load_metric
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoTokenizer, BatchEncoding,
                          get_scheduler, set_seed)
from transformers.modeling_outputs import SequenceClassifierOutput

import wandb
from argparser import parse_args
from evaluation import all_metrics
from modeling_bert import BertForMultilabelClassification
from modeling_longformer import LongformerForMultilabelClassification
from modeling_roberta import RobertaForMultilabelClassification
from modelling_caml import ConvolutionalAttentionPool


MODELS_CLASSES = {
    'bert':       BertForMultilabelClassification,
    'roberta':    RobertaForMultilabelClassification,
    'longformer': LongformerForMultilabelClassification,
    'caml':       ConvolutionalAttentionPool,
}

def main():
    assert torch.cuda.is_available(), "No GPU available!"

    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    
    ### XXX Add this back to load all labels
    # Labels
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique

    if args.code_50l:
        # the set of `train_labels` in `collectrare50data` from the KEPT paper repo.
        # XXX For some reason there are 53 labels lol... whatever I guess.
        labels = {'747.69', '512.2', '959.7', '710.8', '785.9', '955.3', '34.71', '550.11', '955.1;40.7', 
                  '447.9', '40.7', 'V18.3', '719.49', '477.8', '506.0', 'V65.3', '53.02', 'V10.61', '148.1',
                  '780.94', '958.91', '999.82', '252.08', '77.7', '955.1', '701.0', '813.32', '52.0', '816.02',
                  '998.01', '318.2', '351.9', '171.0', '38.47', '607.82', '990', 'V26.52', '338.28', '378.52',
                  '17.36', '453.50', '873.52', '176.0', '596.89', '202.82', '569.42', '282.2', '270.6', '737.43',
                  '790.8', '362.11'}
        logging.warning("Coding the rarest 50 codes with sufficient data to evaluate on %s", labels)
    else:
        if "50l" in args.train_file:
            logging.warning("WARNING: enable --code_50l to train only on the 50 rarest codes!")

        labels = set()
        all_codes_file = "../data/mimic3/ALL_CODES.txt" if not args.code_50 else "../data/mimic3/ALL_CODES_50.txt"
        if args.code_file is not None:
            all_codes_file = args.code_file

        with open(all_codes_file, "r") as f:
            for line in f:
                if line.strip() != "":
                    labels.add(line.strip())

    label_list = sorted(labels)

    if args.only_labels_in_train_set:
        logger.warning("ONLY USING LABELS IN THE TRAINING SET!!!")
        train_labels = set(';'.join(raw_datasets["train"]["LABELS"]).split(";")) # I'm either a genius or a ...
        label_list = [l for l in label_list if l in train_labels]
    ###

    # label_list = ["250.01", "250.02"] # Diabetes types I and II, respectively.

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    try:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    except (AttributeError, EnvironmentError): # CAML model has no 'from_pretrained' attr. Hugginface has no config for our model (we don't care).
        config = AutoConfig.from_pretrained("roberta-base", num_labels=num_labels, finetuning_task=None)
    if args.model_type == "longformer":
        config.attention_window = args.chunk_size
    elif args.model_type in ["bert", "roberta"]:
        config.model_mode = args.model_mode
    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base",
        # args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        do_lower_case=not args.cased)
    model_class = MODELS_CLASSES[args.model_type]
    if args.num_train_epochs > 0:
        if hasattr(model_class, "from_pretrained"):
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
        else:
            # CAML model isn't a pretrained model.
            model: torch.nn.Module = model_class(config=config, conditioning_layer=args.conditioning)

            # if args.conditioning:
            #     # Initialize the conditioning weight to Identity
            #     model._conditioning.weight = torch.nn.Parameter(torch.eye(n=num_labels,
            #                                                               requires_grad=True))
            
            ### XXX IF WE'RE LOADING AND FREEZING THE WEIGHTS
            # x = torch.load(CAML_MODEL_PATH)
            # logging.info("LOADING MODEL WEIGHTS FROM %s", CAML_MODEL_PATH)
            
            # breakpoint()
            # x["_conditioning.weight"] = conditioning_weight
            # x["_conditioning.bias"]   = torch.zeros([num_labels])
            
            # # ###

            # model.load_state_dict(x)
            # for name, param in model.named_parameters():
            #     param.requires_grad = bool("_conditioning" in name)
            ###
    else:
        if hasattr(model_class, "from_pretrained"):
            model = model_class.from_pretrained(
                args.output_dir,
                config=config,
            )
        else:
            # CAML model isn't a pretrained model.
            model = model_class(config=config, conditioning_layer=args.conditioning)

    sentence1_key, sentence2_key = "TEXT", None

    label_to_id: dict[str, int] = {v: i for i, v in enumerate(label_list)}
    """A map of ICD-9 diagnostic codes to positional label indices.
    
    Example:
        Diabetes mellitus without mention of complications.
        { "250.00" : 8515 } 
        Label indexed at 8515.
        e.g if x[8515] == 1:
            'diagnosed with diabetes'
    """
    padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result: BatchEncoding = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True,
                                           add_special_tokens=("cls" not in args.model_mode))

        if "LABELS" in examples:
            result["labels"] = examples["LABELS"]

            ### Extract the labels ###
            label_ids: list[int] = []
            for labels in examples["LABELS"]:
                if labels is None:
                    label_ids.append([]) # Sample has no labels!
                else:
                    sample_labels = []
                    for label in labels.strip().split(';'):
                        if (stripped_label := label.strip()) != "":
                            # XXX This line is necessary if we're looking at a subset of labels!
                            if stripped_label in label_list:
                                sample_labels.append(label_to_id[stripped_label])
                            #####
                    # if len(sample_labels) == 2:
                    #     breakpoint() # these diabetes codes should be mutually exclusive!
                    label_ids.append(sample_labels)
            ##########################
        
            result["label_ids"] = label_ids
        return result

    remove_columns = raw_datasets["train"].column_names if args.train_file is not None else raw_datasets["validation"].column_names
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=remove_columns)

    ### XXX Downsampling no-label background for the training dataset
    # train = processed_datasets["train"]

    # type_1: int = train["label_ids"].count([0])
    # type_2: int = train["label_ids"].count([1])
    # min_samples = min(type_1, type_2)
    # print("Min samples for all label combinations: ", min_samples)

    # # Instances with no labels.
    # x = [0]
    # y = [1]
    # z = [ ] # Empty list, used for comparisons!

    # type_1_indices     = [i for (i, instance) in enumerate(train) if instance["label_ids"] == x]
    # type_2_indices     = [i for (i, instance) in enumerate(train) if instance["label_ids"] == y]
    # background_indices = [i for (i, instance) in enumerate(train) if instance["label_ids"] == z]
    
    # # Subsample the dominant class
    # background_sample_indices = random.sample(background_indices, min_samples)

    # from itertools import chain
    # sample_indices = list(chain(type_1_indices, type_2_indices, background_sample_indices))
    
    # random.shuffle(sample_indices)
    
    # assert len(set(sample_indices)) == len(type_1_indices) + len(type_2_indices) + min_samples, \
    #     "We should sample only a subset of the background indices!"
    
    ###

    eval_dataset = processed_datasets["validation"]

    if args.num_train_epochs > 0:
        train_dataset = processed_datasets["train"]
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            logger.info(f"Original tokens: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    def data_collator(features):
        batch = dict()

        if "cls" in args.model_mode:
            for f in features:
                new_input_ids = []
                for i in range(0, len(f["input_ids"]), args.chunk_size - 2):
                    new_input_ids.extend([tokenizer.cls_token_id] + f["input_ids"][i:i+(args.chunk_size)-2] + [tokenizer.sep_token_id])
                f["input_ids"] = new_input_ids
                f["attention_mask"] = [1] * len(f["input_ids"])
                f["token_type_ids"] = [0] * len(f["input_ids"])

        max_length = max([len(f["input_ids"]) for f in features])
        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size

        batch["input_ids"] = torch.tensor([
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]).contiguous().view((len(features), -1, args.chunk_size))
        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.tensor([
                f["attention_mask"] + [0] * (max_length - len(f["attention_mask"]))
                for f in features
            ]).contiguous().view((len(features), -1, args.chunk_size))
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.tensor([
                f["token_type_ids"] + [0] * (max_length - len(f["token_type_ids"]))
                for f in features
            ]).contiguous().view((len(features), -1, args.chunk_size))
        label_ids = torch.zeros((len(features), len(label_list)))
        for i, f in enumerate(features):
            for label in f["label_ids"]:
                label_ids[i, label] = 1
        batch["labels"] = label_ids
        return batch

    if args.num_train_epochs > 0:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        ### XXX subsample background sampleswith no labels!

        # train_dataloader = DataLoader(
        #     train_dataset, sampler=sample_indices, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        # )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = {"bias", "LayerNorm.weight"}

    if args.disable_base_model_weight_decay:
        assert isinstance(model, RobertaForMultilabelClassification), "base model weight decay only defined for Roberta!"
        no_decay.update(n for n, _p in model.roberta.named_parameters())
        logging.warning("Setting RoBERTa weight decay to 0!")

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, eval_dataloader = accelerator.prepare(
        model, optimizer, eval_dataloader
    )
    if args.num_train_epochs > 0:
        train_dataloader = accelerator.prepare(train_dataloader)

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    # Train!
    wandb.init(project="CAML-per-label-labelsmoothing", config=args,
               name=args.run_name)

    if args.num_train_epochs > 0:
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        early_stopping_tolerance = 5
        eval_f1_scores  = [0.0 for _ in range(early_stopping_tolerance)] # Seed with zeros so we can can do indexing
        train_f1_scores = [0.0 for _ in range(early_stopping_tolerance)] # Seed with zeros so we can can do indexing
        for epoch in tqdm(range(args.num_train_epochs)):
            model.train()
            epoch_loss = 0.0

            all_train_preds = []
            all_train_preds_raw = []
            all_train_labels = []
            for step, batch in enumerate(train_dataloader):
                outputs_train = model(**batch)
                loss = outputs_train.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                epoch_loss += loss.item()
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_postfix(loss=epoch_loss / completed_steps)
                
                    ### train metrics
                    train_preds_raw = outputs_train.logits.sigmoid().cpu().detach()
                    train_preds     = (train_preds_raw > 0.5).int()
                    
                    all_train_preds_raw.extend(list(train_preds_raw))
                    all_train_preds    .extend(list(train_preds))
                    all_train_labels   .extend(list(batch["labels"].cpu().numpy()))
                
                if completed_steps >= args.max_train_steps:
                    break
                
            all_train_preds_raw = np.stack(all_train_preds_raw)
            all_train_preds     = np.stack(all_train_preds)
            all_train_labels    = np.stack(all_train_labels)
            
            train_metrics = all_metrics(yhat=all_train_preds, y=all_train_labels, yhat_raw=all_train_preds_raw)
            logger.info(f"TRAIN metrics: {train_metrics}")

            train_f1_scores.append(train_metrics["f1_macro"])
            ###

            model.eval()
            all_preds     = []
            all_preds_raw = []
            all_labels    = []
            for step, batch in tqdm(enumerate(eval_dataloader)):
                with torch.no_grad():
                    outputs_eval: SequenceClassifierOutput = model(**batch)
                preds_raw = outputs_eval.logits.sigmoid().cpu()
                preds = (preds_raw > 0.5).int()
                all_preds_raw.extend(list(preds_raw))
                all_preds.extend(list(preds))
                all_labels.extend(list(batch["labels"].cpu().numpy()))
            
            all_preds_raw = np.stack(all_preds_raw)
            all_preds = np.stack(all_preds)
            all_labels = np.stack(all_labels)
            eval_metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
            logger.info(f"epoch {epoch} finished")
            logger.info(f"metrics: {eval_metrics}")
            
            eval_f1_scores.append(eval_metrics["f1_macro"])

            ## Log metrics to weights and biases. Requires a flat dict.
            joint_metrics_dict = {f"eval_{key}": value for key, value in eval_metrics.items()} \
                               | {f"train_{key}": value for key, value in train_metrics.items()} \
                               | {"eval_loss": outputs_eval.loss, "train_loss": epoch_loss}
            wandb.log(joint_metrics_dict)
            # if min(eval_f1_scores[ - early_stopping_tolerance: -1]) > eval_f1_scores[-1]:
            #     logging.info("EARLY STOPPING DUE TO MACRO f1 DECREASING\n%s", eval_f1_scores)
            #     logging.info("EARLY STOPPING TOLERANCE = %s", early_stopping_tolerance)
            #     break

    if args.num_train_epochs == 0 and accelerator.is_local_main_process:
        model.eval()
        all_preds = []
        all_preds_raw = []
        all_labels = []
        
        from utils.load_icd9_codes import load_code_descriptions
        code_to_description: dict[str, str] = load_code_descriptions()

        if args.show_attention:
            outputs_dir = "outputs\\"
            for fp in os.listdir(outputs_dir):
                os.remove(outputs_dir + fp)

        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs_eval: SequenceClassifierOutput = model(**batch)
            preds_raw = outputs_eval.logits.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            batch_labels = batch["labels"].cpu().numpy()
            all_labels.extend(list(batch_labels))
            if args.show_attention:
                ### Let's generate some explanations. ###
                predicted_labels_idxs = np.where(preds[0] == 1)[0]
                html_output = ""
                if len(predicted_labels_idxs):
                    for label_idx in predicted_labels_idxs:
                        input_ids  = batch["input_ids"].squeeze(0).flatten() # (512,) - input is chunked into 3 (total seq. length is 512)
                        attentions = outputs_eval.attentions.squeeze(0)      # (8921, 512)
                        sample_labels = batch_labels[0] # NOTE: assuming only one sample in the batch!
                        
                        label_attention = attentions[label_idx]

                        import html

                        from utils.construct_html_of_weights import \
                            overlay_sentence_attention
                        tokens = [html.escape(tokenizer.decode(id_)) for id_ in input_ids]

                        code = label_list[label_idx]
                        normalized_attention = label_attention / max(label_attention)
                        html_output += overlay_sentence_attention(tokens, normalized_attention,
                                                                name='',
                                                                true_label="positive" if sample_labels[label_idx] else "negative",
                                                                prediction=preds_raw[0][label_idx],
                                                                label_description=code_to_description.get(code, "no description found"))

                    with open(f"{outputs_dir}patient={step}_pred_labels={len(predicted_labels_idxs)}_true_labels={int(sample_labels.sum())}.html", "w", encoding="utf-8") as f:
                        f.write(html_output)

        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        eval_metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)

        logger.info(f"evaluation finished")
        logger.info(f"metrics: {eval_metrics}")
        wandb.log(eval_metrics)

        for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            all_preds = (all_preds_raw > t).astype(int)
            eval_metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5,8,15])
            logger.info(f"metrics for threshold {t}: {eval_metrics}")

    if args.output_dir is not None and args.num_train_epochs > 0:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), args.output_dir + "\\model_weights.pth")
            print(f"CAML MODEL WEIGHTS SAVED TO {args.output_dir}")

if __name__ == "__main__":
    main()
