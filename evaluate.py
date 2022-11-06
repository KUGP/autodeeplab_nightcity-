import os
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from mypath import Path
from utils.utils import AverageMeter, inter_and_union
from config_utils.evaluate_args import obtain_evaluate_args
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from dataloaders.datasets.cityscapes import CityscapesSegmentation
from dataloaders.datasets.nightcity import NightCitySegmentation
from utils.summaries import TensorboardSummary
import dataloaders


def main(start_epoch, epochs):
    assert torch.cuda.is_available(), NotImplementedError('No cuda available ')
    if not osp.exists('data/'):
        os.mkdir('data/')
    if not osp.exists('log/'):
        os.mkdir('log/')
    args = obtain_evaluate_args()
    torch.backends.cudnn.benchmark = True
    summary = TensorboardSummary('./log')
    writer = summary.create_summary()
    model_fname = 'run/{}/{}_epoch%d.pth'.format(args.dataset,args.checkname)
    if args.dataset == 'cityscapes':
        dataset = CityscapesSegmentation(args=args, root=Path.db_root_dir(args.dataset), split='reval')
    elif args.dataset =='nightcity':
        dataset = NightCitySegmentation(args=args, root=Path.db_root_dir(args.dataset), split='test')
        train_loader, val_loader, test_loader, num_classes = dataloaders.make_data_loader(args)
        args.num_classes = dataset.NUM_CLASSES
    else:
        return NotImplementedError
    if args.backbone == 'autodeeplab':
        model = Retrain_Autodeeplab(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))
    if not args.train:
        val_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        model = torch.nn.DataParallel(model).cuda()
        print("======================start evaluate=======================")
        for epoch in range(epochs):
            print("evaluate epoch {:}".format(epoch))
            checkpoint_name = 'run/nightcity/NC_train5_epoch491.pth'
            print(checkpoint_name)
            checkpoint = torch.load(checkpoint_name)
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
            model.module.load_state_dict(state_dict)
            inter_meter = AverageMeter()
            union_meter = AverageMeter()
            for i, sample in enumerate(test_loader):
                inputs, target = sample['image'], sample['label']
                N, H, W = target.shape
                total_outputs = torch.zeros((N, dataset.NUM_CLASSES, H, W)).cuda()
                with torch.no_grad():
                    for j, scale in enumerate(args.multi_scale):
                        new_scale = [int(H * scale), int(W * scale)]
                        inputs = F.upsample(inputs, new_scale, mode='bilinear', align_corners=True)
                        inputs = inputs.cuda()
                        outputs = model(inputs)
                        outputs = F.upsample(outputs, (H, W), mode='bilinear', align_corners=True)
                        total_outputs += outputs
                    _, pred = torch.max(total_outputs, 1)
                    summary.visualize_image(writer, args.dataset, inputs, target, pred, i + 1)
                    pred = pred.detach().cpu().numpy().squeeze().astype(np.uint8)
                    mask = target.numpy().astype(np.uint8)
                    print('eval: {0}/{1}'.format(i + 1, len(test_loader)))

                    inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
                    inter_meter.update(inter)
                    union_meter.update(union)
            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            miou = 'epoch: {0} Mean IoU: {1:.2f}'.format(epoch, iou.mean() * 100)
            f = open('log/result.txt', 'a')
            for i, val in enumerate(iou):
                class_iou = 'IoU {0}: {1:.2f}\n'.format(dataset.CLASSES[i], val * 100)
                f.write(class_iou)
            f.write('\n')
            f.write(miou)
            f.write('\n')
            f.close()


if __name__ == "__main__":
    epochs = 100
    state_epochs = 1
    main(epochs=epochs, start_epoch=state_epochs)
