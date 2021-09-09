import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from utils.vis_utils import plot_single_pcd
from utils.train_utils import *
from dataset import ShapeNetH5


def test():
    dataset_test = ShapeNetH5(train=False, npoints=args.num_points)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    test_loss_cat = torch.zeros([16, 4], dtype=torch.float32).cuda()
    cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 150 * 26
    novel_cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 50 * 26
    cat_num = torch.cat((cat_num, novel_cat_num), dim=0)
    cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft', 
                'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']
    idx_to_plot = [i for i in range(0, 1600, 75)]

    logging.info('Testing...')
    if args.save_vis:
        save_gt_path = os.path.join(log_dir, 'pics', 'gt')
        save_partial_path = os.path.join(log_dir, 'pics', 'partial')
        save_completion_path = os.path.join(log_dir, 'pics', 'completion')
        os.makedirs(save_gt_path, exist_ok=True)
        os.makedirs(save_partial_path, exist_ok=True)
        os.makedirs(save_completion_path, exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            
            label, inputs_cpu, gt_cpu = data
            # mean_feature = None

            inputs = inputs_cpu.float().cuda()
            gt = gt_cpu.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            # result_dict = net(inputs, gt, is_training=False, mean_feature=mean_feature)
            result_dict = net(inputs, gt, is_training=False)
            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            for j, l in enumerate(label):
                for ind, m in enumerate(metrics):
                    test_loss_cat[int(l), ind] += result_dict[m][int(j)]

            if i % args.step_interval_to_print == 0:
                logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

            if args.save_vis:
                for j in range(args.batch_size):
                    idx = i * args.batch_size + j
                    if idx in idx_to_plot:
                        pic = 'object_%d.png' % idx
                        plot_single_pcd(result_dict['out2'][j].cpu().numpy(), os.path.join(save_completion_path, pic))
                        plot_single_pcd(gt_cpu[j], os.path.join(save_gt_path, pic))
                        plot_single_pcd(inputs_cpu[j].cpu().numpy(), os.path.join(save_partial_path, pic))

        logging.info('Loss per category:')
        category_log = ''
        for i in range(16):
            category_log += '\ncategory name: %s' % (cat_name[i])
            for ind, m in enumerate(metrics):
                scale_factor = 1 if m == 'f1' else 10000
                category_log += ' %s: %f' % (m, test_loss_cat[i, ind] / cat_num[i] * scale_factor)
        logging.info(category_log)

        logging.info('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        logging.info(overview_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test()
