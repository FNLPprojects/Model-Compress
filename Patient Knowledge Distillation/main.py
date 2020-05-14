import logging
import os
import random
import pickle
import argparse
import json

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn
from tqdm import tqdm, trange

from transformers import BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup


from src.data_processor import get_task_dataloader, processors, output_modes
from src.model import BertForSequenceClassificationEncoder, FCClassifierForSequenceClassification
from src.utils import load_model, count_parameters, eval_model_dataloader, compute_metrics
from src.KD_loss import distillation_loss, patience_loss

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataloader, encoder, classifier, tokenizer):
    num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    param_optimizer = list(encoder.named_parameters()) + list(classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=args.learning_rate,eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    global_step = 0
    encoder.train()
    classifier.train()

    for epoch in tqdm(range(args.num_train_epochs), desc="Epoch"):
        tr_loss, logging_loss = 0.0, 0.0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(args.device) for t in batch)
            if args.alpha == 0:
                input_ids, input_mask, segment_ids, label_ids = batch
                teacher_pred, teacher_patience = None, None
            else:
                input_ids, input_mask, segment_ids, label_ids, teacher_pred, teacher_patience = batch

            # define a new function to compute loss values for both output_modes
            full_output, pooled_output = encoder(input_ids, segment_ids, input_mask)
            logits_pred_student = classifier(pooled_output)
            student_patience = torch.stack(full_output[:-1]).transpose(0, 1)

            kd_loss = distillation_loss(logits_pred_student, label_ids, teacher_pred, T=args.T, alpha=args.alpha)
            if args.beta > 0:
                if student_patience.shape[0] != input_ids.shape[0]:
                    n_layer = student_patience.shape[1]
                    student_patience = student_patience.transpose(0, 1).contiguous().view(n_layer, input_ids.shape[0], -1).transpose(0,1)
                pt_loss = args.beta * patience_loss(teacher_patience, student_patience, args.normalize_patience)
                loss = kd_loss + pt_loss
            else:
                loss = kd_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            n_sample = input_ids.shape[0]
            tr_loss += loss.item() * n_sample

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{'step': global_step}}))

        eval_examples, eval_dataloader, eval_label_ids = get_task_dataloader(args, 'dev', tokenizer,
                                                                             SequentialSampler,
                                                                             batch_size=args.eval_batch_size)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        result = evaluate(args, eval_label_ids, encoder, classifier,eval_dataloader)
        logger.info('epoch:{},acc:{},eval_loss:{}'.format(epoch+1, result['acc'], result['eval_loss']))

        if args.n_gpu > 1:
            torch.save(encoder.module.state_dict(), os.path.join(args.output_dir, f'{args.train_type}_epoch{epoch}.encoder.pkl'))
            torch.save(classifier.module.state_dict(), os.path.join(args.output_dir, f'{args.train_type}_epoch{epoch}.cls.pkl'))
        else:
            torch.save(encoder.state_dict(), os.path.join(args.output_dir, f'{args.train_type}_epoch{epoch}.encoder.pkl'))
            torch.save(classifier.state_dict(), os.path.join(args.output_dir, f'{args.train_type}_epoch{epoch}.cls.pkl'))
    return global_step, tr_loss/global_step

def evaluate(args, eval_label_ids, encoder, classifier, dataloader):
    encoder.eval()
    classifier.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        label_ids = label_ids.to(args.device)

        with torch.no_grad():
            full_output, pooled_output = encoder(input_ids, segment_ids, input_mask)
            logits = classifier(pooled_output)

        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(args.task_name, preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    return result



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        help="The name of the task for training.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str,
                        help="student bert model configuration folder")
    parser.add_argument("--encoder_checkpoint",
                        default=None,
                        type=str,
                        help="check point for student encoder")
    parser.add_argument("--cls_checkpoint",
                        default=None,
                        type=str,
                        help="check point for student classifier")
    parser.add_argument("--alpha",
                        default=0.95,
                        type=float,
                        help="alpha for distillation")
    parser.add_argument("--T",
                        default=10.,
                        type=float,
                        help="temperature for distillation")
    parser.add_argument("--beta",
                        default=0.0,
                        type=float,
                        help="weight for AT loss")
    parser.add_argument("--fc_layer_idx",
                        default=None,
                        type=str,
                        help="layers ids we will put FC layers on")
    parser.add_argument("--normalize_patience",
                        default=False,
                        help="normalize patience or not")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="do training or not")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="do evaluation during training or not")

    parser.add_argument("--train_type", default="finetune_teacher",
                        choices=["finetune_teacher","train_student"],
                        help="choose which to train")
    parser.add_argument("--log_every_step",
                        default=50,
                        type=int,
                        help="output to log every global x training steps, default is 1")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--logging_steps',
                        type=int,
                        default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--student_hidden_layers',
                        type=int,
                        default=12,
                        help="number of transformer layers for student, default is None (use all layers)")
    parser.add_argument('--teacher_prediction',
                        type=str,
                        default=None,
                        help="teacher prediction file to guild the student's output")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    logger.info('actual batch size on all GPU = %d' % args.train_batch_size)

    if args.train_type == 'finetune_teacher':
        args.student_hidden_layers = 12 if 'base' in args.bert_model else 24
        args.alpha = 0.0   # alpha = 0 is equivalent to fine-tuning for KD
    elif args.train_type == "train_student":
        args.student_hidden_layers = 6
        args.kd_model = "kd.cls"
        args.alpha = 0.7
        args.beta = 500
        args.T = 10
        args.fc_layer_idx = "1,3,5,7,9"   # this for pkd-skip
        args.normalize_patience = True
    else:
        raise ValueError("please pick train_type from finetune_teacher,train_student")

    if args.encoder_checkpoint is None:
        args.encoder_checkpoint = os.path.join(args.bert_model, 'pytorch_model.bin')
        logger.info('encoder checkpoint not provided, use pre-trained at %s instead' % args.encoder_checkpoint)

    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))


    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    #args.n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(args.device, args.n_gpu))

    # set seed
    set_seed(args)

    # prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    # prepare tokenizer and model
    config = BertConfig()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    config.output_hidden_states = True

    encoder = BertForSequenceClassificationEncoder(config, num_hidden_layers=args.student_hidden_layers)
    classifier = FCClassifierForSequenceClassification(config, args.num_labels, config.hidden_size, 0)

    n_student_layer = len(encoder.bert.encoder.layer)
    encoder = load_model(encoder, args.encoder_checkpoint, args, 'student', verbose=True)
    logger.info('*' * 77)
    classifier = load_model(classifier, args.cls_checkpoint, args, 'classifier', verbose=True)


    n_param_student = count_parameters(encoder) + count_parameters(classifier)
    logger.info('number of layers in student model = %d' % n_student_layer)
    logger.info('num parameters in student model are %d' % n_param_student)

    # Training
    if args.do_train:
        read_set = 'train'
        if args.train_type == "train_student":
            assert args.teacher_prediction is not None
            assert args.alpha > 0
            logger.info('loading teacher\'s predictoin')
            teacher_predictions = pickle.load(open(args.teacher_prediction, 'rb'))['train'] if args.teacher_prediction is not None else None
            logger.info('teacher acc = %.2f, teacher loss = %.5f' % (
            teacher_predictions['acc'] * 100, teacher_predictions['loss']))
            train_examples, train_dataloader, _ = get_task_dataloader(args, read_set, tokenizer,
                                                                      SequentialSampler,
                                                                      batch_size=args.train_batch_size,
                                                                      knowledge=teacher_predictions['pred_logit'],
                                                                      extra_knowledge=teacher_predictions[
                                                                          'feature_maps'])
        else:
            assert args.alpha == 0
            logger.info("runing teacher fine-tuning")
            train_examples, train_dataloader, _ = get_task_dataloader(args, read_set, tokenizer,
                                                                      SequentialSampler,
                                                                      batch_size=args.train_batch_size)

        global_step, tr_loss = train(args, train_dataloader, encoder, classifier, tokenizer)
        #################
        # information of teacher model (like [CLS])
        #################
        if args.train_type == "finetune_teacher":
            all_res = {'train': None}

            encoder_file = os.path.join(args.output_dir,f'{args.train_type}_epoch{args.num_train_epochs-1}.encoder.pkl')
            cls_file = os.path.join(args.output_dir,f'{args.train_type}_epoch{args.num_train_epochs-1}.cls.pkl')
            print("encoder_file")

            encoder = BertForSequenceClassificationEncoder(config, num_hidden_layers=args.student_hidden_layers)
            classifier = FCClassifierForSequenceClassification(config, args.num_labels, config.hidden_size, 0)

            encoder = load_model(encoder, encoder_file, args, 'exact', verbose=True)
            classifier = load_model(classifier, cls_file, args, 'exact', verbose=True)
            
            train_res = eval_model_dataloader(encoder, classifier, train_dataloader, args.device, detailed=True,
                                              verbose=False)
            all_res['train'] = train_res

            logger.info('saving teacher results')

            fname = os.path.join(args.output_dir,
                                 args.task_name + f'_teacher_{args.student_hidden_layers}layer_information.pkl')
            with open(fname, 'wb') as fp:
                pickle.dump(all_res, fp)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    if args.do_eval:


        test_examples, test_dataloader, test_label_ids = get_task_dataloader(args, 'dev', tokenizer,
                                                                             SequentialSampler,
                                                                             batch_size=args.eval_batch_size)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        result = evaluate(args, test_label_ids, encoder,classifier,test_dataloader)

        output_test_file = os.path.join(args.output_dir, "test_results_" + '.txt')
        with open(output_test_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return


if __name__ == "__main__":
    main()







