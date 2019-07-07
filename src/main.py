import argparse
import datetime
import logging
import pathlib
import random
import socket
import sys

import torch
import torch.distributed as dist 
import torch.multiprocessing as mp 
import numpy as np 

from trainer import Trainer 

def get_exp_path():
	'''Retrun new experiment path.'''
	return '../log/exp-{0}'.format(
		datetime.datetime.now().strftime('%m-%d-%H:%M:%S'))


def get_logger(path, rank=None):
	'''Get logger for experiment.'''
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

	if rank is None:
		formatter = logging.Formatter('%(asctime)s-%(message)s')
	else:
		formatter = logging.Formatter('%(asctime)s - [worker '
			+ str(rank) +'] - %(message)s')
	
	# stderr log
	handler = logging.StreamHandler(sys.stderr)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	# file log
	handler = logging.FileHandler(path)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	return logger


def worker(rank, args):
	logger = get_logger(args.path + '/experiment.log',
						rank) # process specific logger
	args.logger = logger
	args.rank = rank
	dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % args.port,
		world_size=args.gpus, rank=args.rank)

	# seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	trainer = Trainer(args)

	if args.img1 is not None and args.img2 is not None and args.seg1 is not None and args.seg2 is not None:
		trainer.model.eval()
		trainer.eval_generate_sequence(args.img1, args.img2, args.seg1, args.seg2)
		return
	
	if args.validate:
  		trainer.validate()
	else:
		if args.rank == 0:
			pathlib.Path('../checkpoint').mkdir(
					parents=True, exist_ok=False)

		for epoch in range(args.epochs):
			trainer.set_epoch(epoch)
			trainer.train()
			metrics = trainer.validate()

			if args.rank == 0:	# gpu id
				trainer.save_checkpoint(metrics)


def main():
	parser = argparse.ArgumentParser(description='Train a segmentation completion network')
	parser.add_argument('-d', '--dataset', type=str, default='cityscape',
						help='training dataset', choices=['cityscape'])
	parser.add_argument('--train_dir', type=str,
						default='/data/agong/train', help='Cityscape train dir')
	parser.add_argument('--val_dir', type=str,
						default='/data/agong/val', help='Cityscape val dir')
	parser.add_argument('--test_dir', type=str,
						default='/data/agong/test', help='Cityscape test dir')
	parser.add_argument('--validate', action='store_true',
						help='whether eval after each training')
	parser.add_argument('--val_interval', dest='val_interval',
						help='number of epochs to evaluate',type=int,default=1)
	parser.add_argument('-a', '--arch', type=str, default='CoordGridNet', help='model to use',
						choices=['GridNet','CoordGridNet'])
	parser.add_argument('-bs','--batch_size', type=int,
						default=32, help='Batch size (over multiple gpu)')
	parser.add_argument('-e', '--epochs', type=int,
						default=10, help='Number of training epochs')
	parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
	parser.add_argument('--img1', type=str, default=None, help='First image url')
	parser.add_argument('--img2', type=str, default=None, help='Second image url')
	parser.add_argument('--seg1', type=str, default=None, help='First image seg url')
	parser.add_argument('--seg2', type=str, default=None, help='Second image seg url')
	# parser.add_argument('-emb', '--embedding_dim', type=int,
	# 					default=15, help="embedding dimension")
	# idstributed training
	parser.add_argument('-j', '--workers', type=int, default=4,
						help='Number of data loading workers')
	parser.add_argument('--port', type=int, default=None, help='Port for distributed training')
	parser.add_argument('--seed', type=int, default=1024, help='Random seed')
	parser.add_argument('--print_freq', type=int,
						default=10, help='Print frequency')

	parser.add_argument('--cfg', help='Experiment configure file name', 
		default='./seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
	parser.add_argument('opts', help="Modify config options using the command-line",
						default=None, nargs=argparse.REMAINDER)
	# save and load
	parser.add_argument('-p', '--path', type=str,
						default=None, help='Experiment path')
	parser.add_argument('--ckpt', type=str, default=None,
						help='Path to checkpoint to load')
	# Lin
	parser.add_argument('--start_epoch', dest='start_epoch',
											help='starting epoch',
											default=1, type=int)
	parser.add_argument('--disp_interval', dest='disp_interval',
											help='number of iterations to display',
											default=10, type=int)
	# config optimization
	parser.add_argument('--o', dest='optimizer', help='training optimizer',
						choices =['adamax','adam', 'sgd'], default="adamax")
	parser.add_argument('--lr', dest='lr', help='starting learning rate',
						default=0.001, type=float)
	parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='step to do learning rate decay, unit is epoch',
						default=5, type=int)
	parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
						help='learning rate decay ratio', default=0.1, type=float)

	args = parser.parse_args()
	
	# exp path
	if args.path is None:
		args.path = get_exp_path()
	pathlib.Path(args.path).mkdir(parents=True, exist_ok=False)
	(pathlib.Path(args.path) / 'checkpoint').mkdir(parents=True, exist_ok=False)

	# find free port
	if args.port is None:
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.bind(('', 0))
			s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			args.port = int(s.getsockname()[1])

	# logger
	logger = get_logger(args.path + '/experiment.log')
	logger.info('Start of experiment')
	logger.info('=========== Initilized logger =============')
	logger.info('\n\t' + '\n\t'.join('%s: %s' % (k, str(v))
		for k, v in sorted(dict(vars(args)).items())))
	
	# distributed training
	args.gpus = torch.cuda.device_count()
	logger.info('Total number of gpus: %d' % args.gpus)
	mp.spawn(worker, args=(args,), nprocs=args.gpus)

if __name__ == '__main__':
	main()
