def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),'iou': AverageMeter(),'dice_coef':AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice_score = dice_coef(output,target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice_coef'].update(dice_score, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice_coef',avg_meters['dice_coef'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),('iou', avg_meters['iou'].avg),('dice_coef', avg_meters['dice_coef'].avg)])
