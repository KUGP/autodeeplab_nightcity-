import os
import pdb
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim

import dataloaders
from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args
from utils.summaries import TensorboardSummary
from utils.saver import Saver
from utils.metrics import Evaluator




def main():

    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    args = obtain_retrain_autodeeplab_args()
    saver = Saver(args)
    # saver.save_experiment_config()

    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()

    model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(args.backbone, args.dataset, args.exp)
    if args.dataset == 'pascal':
        raise NotImplementedError
    elif args.dataset == 'cityscapes':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = num_classes

    elif args.dataset == 'nightcity':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = num_classes
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone == 'autodeeplab':
        model = Retrain_Autodeeplab(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.criterion == 'Ohem':
        args.thresh = 0.7
        args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
        args.n_min = int((args.batch_size / len(args.gpu) * args.crop_size[0] * args.crop_size[1]) // 16)
    criterion = build_criterion(args)

    evaluator = Evaluator(num_classes)
    print("##########################Model########################")
    print(model)
    print("#######################################################")

    model = nn.DataParallel(model).cuda()
    model.train()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    optimizer = optim.SGD(model.module.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iteration = len(dataset_loader) * args.epochs
    scheduler = Iter_LR_Scheduler(args, max_iteration, len(dataset_loader))
    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('=> no checkpoint found at {0}'.format(args.resume))


    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        num_img_tr = len(dataset_loader)
        losses = AverageMeter()
        evaluator.reset()
        for i, sample in enumerate(dataset_loader):

            # print("len dataset_loader",len(dataset_loader))   187 for nightcity with batch_size 16

            cur_iter = epoch * len(dataset_loader) + i
            scheduler(optimizer, cur_iter)
            inputs = sample['image'].cuda()
            target = sample['label'].cuda()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            epoch_loss += loss.item()
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            losses.update(loss.item(), args.batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(target.size())
            # print(outputs.size())
            pred = outputs.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            targets = target.cpu().numpy()

            evaluator.add_batch(targets, pred)

            # # Show 10 * 3 inference results each epoch
            # if i % (num_img_tr // 10) == 0:
            #     global_step = i + num_img_tr * epoch
            #     summary.visualize_image(writer, args.dataset, inputs, target, outputs, global_step)




            print('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                epoch + 1, i + 1, len(dataset_loader), scheduler.get_lr(optimizer), loss=losses))

        if epoch < args.epochs - 50:
            is_best = False
            if epoch % 50 == 0:
                # torch.save({
                #     'epoch': epoch + 1,
                #     'state_dict': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                # }, model_fname % (epoch + 1))

                saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best)
        else:
            is_best = False
            # torch.save({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            # }, model_fname % (epoch + 1))

            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best)

        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        writer.add_scalar('train/total_loss_epoch',epoch_loss/len(dataset_loader), epoch)
        writer.add_scalar('train/mIoU', mIoU, epoch)
        writer.add_scalar('train/Acc', Acc, epoch)
        writer.add_scalar('train/Acc_class', Acc_class, epoch)
        writer.add_scalar('train/fwIoU', FWIoU, epoch)
        summary.visualize_image(writer, args.dataset, inputs, target, outputs, epoch)
        # summary.visualize_image(writer, args.dataset, inputs, target, outputs, epoch)



        print('reset local total loss!')
    ### validation code


    writer.close()



if __name__ == "__main__":
    main()
