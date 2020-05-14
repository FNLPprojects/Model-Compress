import logging
import torch
import os

import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def count_parameters(model, trainable_only=True, is_dict=False):
    if is_dict:
        return sum(np.prod(list(model[k].size())) for k in model)
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def load_model(model, checkpoint, args, mode='exact', verbose=True):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param verbose:
    :return:
    """

    n_gpu = args.n_gpu
    device = args.device
    if checkpoint is None:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint))
        model_state_dict = torch.load(checkpoint)
        old_keys = []
        new_keys = []
        for key in model_state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict[new_key] = model_state_dict.pop(old_key)

        del_keys = []
        keep_keys = []
        if mode == 'exact':
            pass
        elif mode == 'classifier':
            for t in list(model_state_dict.keys()):
                if 'classifier' not in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'student':
            model_keys = model.state_dict().keys()
            for t in list(model_state_dict.keys()):
                if t not in model_keys:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        else:
            raise ValueError('%s not available for now' % mode)
        model.load_state_dict(model_state_dict)
        if mode != 'exact':
            logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model


def eval_model_dataloader(encoder_bert, classifier, dataloader, device, detailed=False,
                          criterion=nn.CrossEntropyLoss(reduction='sum'), use_pooled_output=True,
                          verbose = False):
    """
    :param encoder_bert:  either a encoder, or a encoder with classifier
    :param classifier:    if a encoder, classifier needs to be provided
    :param dataloader:
    :param device:
    :param detailed:
    :return:
    """
    if hasattr(encoder_bert, 'module'):
        encoder_bert = encoder_bert.module
    if hasattr(classifier, 'module'):
        classifier = classifier.module

    n_layer = len(encoder_bert.bert.encoder.layer)
    encoder_bert.eval()
    if classifier is not None:
        classifier.eval()

    loss = 0
    acc = 0

    # set loss function
    if detailed:
        feature_maps = [[] for _ in range(n_layer)]   # assume we only deal with bert base here
        predictions = []
        pooled_feat_maps = []

    for idx, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        if len(batch) > 4:
            input_ids, input_mask, segment_ids, label_ids, *ignore = batch
        else:
            input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            if classifier is None:
                preds = encoder_bert(input_ids, segment_ids, input_mask)
            else:
                feat = encoder_bert(input_ids, segment_ids, input_mask)
                if isinstance(feat, tuple):
                    feat, pooled_feat = feat
                    if use_pooled_output:
                        preds = classifier(pooled_feat)
                    else:
                        preds = classifier(feat)
                else:
                    feat, pooled_feat = None, feat
                    preds = classifier(pooled_feat)
        loss += criterion(preds, label_ids).sum().item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(label_ids).sum().cpu().item()

        if detailed:
            bs = input_ids.shape[0]
            need_reshape = bs != pooled_feat.shape[0]
            if classifier is None:
                raise ValueError('without classifier, feature cannot be calculated')
            if feat is None:
                pass
            else:
                for fm, f in zip(feature_maps, feat):
                    if need_reshape:
                        fm.append(f.contiguous().view(bs, -1).detach().cpu().numpy())
                    else:
                        fm.append(f.detach().cpu().numpy())
            if need_reshape:
                pooled_feat_maps.append(pooled_feat.contiguous().view(bs, -1).detach().cpu().numpy())
            else:
                pooled_feat_maps.append(pooled_feat.detach().cpu().numpy())

            predictions.append(preds.detach().cpu().numpy())
        if verbose:
            logger.info('input_ids.shape = {}, tot_loss = {}, tot_correct = {}'.format(input_ids.shape, loss, acc))

    loss /= len(dataloader.dataset) * 1.0
    acc /= len(dataloader.dataset) * 1.0
    if detailed:
        feat_maps = [np.concatenate(t) for t in feature_maps] if len(feature_maps[0]) > 0 else None
        if n_layer == 24:
            return {'loss': loss,
                    'acc': acc,
                    'pooled_feature_maps': np.concatenate(pooled_feat_maps),
                    'pred_logit': np.concatenate(predictions),
                    'feature_maps': [feat_maps[i] for i in [3, 7, 11, 15, 19]]}
        else:
            return {'loss': loss,
                    'acc': acc,
                    'pooled_feature_maps': np.concatenate(pooled_feat_maps),
                    'pred_logit': np.concatenate(predictions),
                    'feature_maps': feat_maps}

    return {'loss': loss, 'acc': acc}




