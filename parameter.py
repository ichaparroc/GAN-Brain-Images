import argparse

#def str2bool(v):
#    return v.lower() in ('true')

def get_parameters():

	parser = argparse.ArgumentParser()

	# Model hyper-parameters

	parser.add_argument('--dataset', type=str, default='brats2019', choices=['basel', 'isles2018', 'brats2019'])
	parser.add_argument('--task', type=str, default='segmentation', choices=['segmentation', 'synthesis'])

	parser.add_argument('--type_label',	type=str, default='label',choices=['label', 'label_input'])

	parser.add_argument('--data_path', type=str, default='../')

	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--sample_size',type=int, default=3)
	parser.add_argument('--lr', type=float, default=0.0002)
	parser.add_argument('--beta1', type=float, default=0.5)
	parser.add_argument('--beta2', type=float, default=0.999)
	
	#parser.add_argument('--z_dim', type=int, default=128)
	
	#TODO
	#parser.add_argument('--g_conv_dim', type=int, default=64)
	#parser.add_argument('--d_conv_dim', type=int, default=64)

	#TODO
	#parser.add_argument('--g_lr', type=float, default=0.0001)
	#parser.add_argument('--d_lr', type=float, default=0.0004)

	#parser.add_argument('--lr_decay', type=float, default=0.95)


	return parser.parse_args()
