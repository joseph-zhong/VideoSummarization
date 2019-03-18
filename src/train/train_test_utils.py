"""
train_test_utils.py
---

Training and Test-time Utilities.
"""

import src.utils.coco as _coco
import src.data.msrvtt as _msrvtt

from extern.coco_caption.pycocotools.coco import COCO
from extern.coco_caption.pycocoevalcap.eval import COCOEvalCap

def train_step():
    pass

def eval_step(eval_loader, banet, prediction_txt_path, reference, use_cuda=False):
    result = {}
    for i, (videos, video_ids) in enumerate(eval_loader):
        if use_cuda:
            videos = videos.cuda()

        outputs, _ = banet(videos, None)
        for (tokens, vid) in zip(outputs, video_ids):
            s = banet.decoder.decode_tokens(tokens.data)
            result[vid] = s

    prediction_txt = open(prediction_txt_path, 'w')
    for vid, s in result.items():
        prediction_txt.write('%d\t%s\n' % (vid, s))  # 注意，MSVD数据集的视频文件名从1开始

    prediction_txt.close()

    # 开始根据生成的结果计算各种指标
    metrics = measure(prediction_txt_path, reference)
    return metrics

def measure(prediction_txt_path, reference):
    # 把txt格式的预测结果转换成检验程序所要求的格式
    crf = _coco.CocoResFormat()
    crf.read_file(prediction_txt_path, True)

    # crf.res就是格式转换之后的预测结果
    cocoRes = reference.loadRes(crf.res)
    cocoEval = COCOEvalCap(reference, cocoRes)

    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score))
    return cocoEval.eval
