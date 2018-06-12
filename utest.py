from networks import create_Generator, BasicBlock, ResBlock
import torch
import torch.nn as nn 
from torch.autograd import Variable
import pytest, random


def assert_configs(g, res, epc, dpc, init):
	assert g.learning_residual == res
	assert g.encoder_partial_conv == epc
	assert g.decoder_partial_conv == dpc 
	assert g.init_type == init

@pytest.mark.parametrize('block', ['basic', 'residual'])
def test_create_generator(block):
	x = Variable(torch.randn(1, 3, 512, 512))
	mask = Variable(torch.bernoulli(torch.rand(1, 1, 512, 512)))

	# only conv
	print('only conv')
	config = {'g_input': 'masked_X', 'decoder_partial_conv': False, 'init_type': 'kaiming', \
			'learning_residual': False, 'block': block, 'progressive_growing': False, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, False, False, False, 'kaiming')
	out1 = g(x)
	assert out1.size() == x.size()

	# only partial conv on encoder
	print('only partial conv on encoder')
	config = {'g_input': 'masked_X+mask', 'decoder_partial_conv': False, 'init_type': 'kaiming', \
			'learning_residual': False, 'block': block, 'progressive_growing': False, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, False, True, False, 'kaiming')
	out2 = g(x, mask)
	assert out2.size() == x.size()

	# full partial conv
	print('full partial conv')
	config = {'g_input': 'masked_X+mask', 'decoder_partial_conv': True, 'init_type': 'kaiming', \
			'learning_residual': False, 'block': block, 'progressive_growing': False, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, False, True, True, 'kaiming')
	out3 = g(x, mask)
	assert out3.size() == x.size()

	# conv + learning residual
	print('conv + learning residual')
	config = {'g_input': 'masked_X', 'decoder_partial_conv': False, 'init_type': 'kaiming', \
			'learning_residual': True, 'block': block, 'progressive_growing': False, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, True, False, False, 'kaiming')
	out4 = g(x)
	assert out4.size() == x.size()

	# only partial conv on encoder + learning residual
	print('only partial conv on encoder + learning residual')
	config = {'g_input': 'masked_X+mask', 'decoder_partial_conv': False, 'init_type': 'kaiming', \
			'learning_residual': True, 'block': block, 'progressive_growing': False, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, True, True, False, 'kaiming')
	out5 = g(x, mask)
	assert out5.size() == x.size()

	# full partial conv + learning residual
	print('full partial conv + learning residual')
	config = {'g_input': 'masked_X+mask', 'decoder_partial_conv': True, 'init_type': 'kaiming', \
			'learning_residual': True, 'block': block, 'progressive_growing': False, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, True, True, True, 'kaiming')
	out6 = g(x, mask)
	assert out6.size() == x.size()


def assert_block_contains(block, module_names):
	_module_names = module_names.copy()
	def check_module_in(m):
		name = m.__class__.__name__
		if name in _module_names:
			_module_names.remove(name)

	block.apply(check_module_in)
	assert len(_module_names) == 0

def assert_block_not_contains(block, module_names):
	def check_module_not_in(m):
		name = m.__class__.__name__
		assert name not in module_names

	block.apply(check_module_not_in)

def assert_size_match(out_size, gt_size):
	assert len(out_size) == len(gt_size)
	assert all([out_size[i]==gt_size[i] for i in range(len(out_size))])

def test_basic_block():
	block = BasicBlock
	x = Variable(torch.randn(1, 3, 64, 64))
	mask = Variable(torch.bernoulli(torch.rand(1, 1, 64, 64)))
	mask2 = Variable(torch.bernoulli(torch.rand(1, 1, 128, 128)))

	# without upsample
	cfg = {'in_channels': 3, 'out_channels': 4, 'kernel_size': 3, 'stride': 2, 'padding': 1}
	# conv-bn-relu
	print('conv-bn-relu')
	b1 = block(False, False, False, **cfg)
	out1 = b1(x)
	assert_block_contains(b1, ['Conv2d', 'BatchNorm2d', 'ReLU'])
	assert_block_not_contains(b1, ['Upsample'])
	assert_size_match(out1.size(), [1, 4, 32, 32])

	# conv-relu
	print('conv-relu')
	b2 = block(False, False, True, **cfg)
	out2 = b2(x)
	assert_block_contains(b2, ['Conv2d', 'ReLU'])
	assert_block_not_contains(b2, ['BatchNorm2d', 'Upsample'])
	assert_size_match(out2.size(), [1, 4, 32, 32])

	# pconv-in-lrelu
	print('pconv-in-lrelu')
	b3 = block(False, True, False, **cfg, norm=nn.InstanceNorm2d, activation=nn.LeakyReLU(0.2))
	out3, _mask_ = b3(x, mask)
	assert_block_contains(b3, ['PartialConv2d', 'InstanceNorm2d', 'LeakyReLU'])
	assert_block_not_contains(b3, ['BatchNorm2d', 'ReLU', 'Upsample'])
	assert_size_match(out3.size(), [1, 4, 32, 32]) and assert_size_match(_mask_.size(), [1, 1, 32, 32])

	# pconv-sigmoid
	print('pconv-sigmoid')
	b4 = block(False, True, True, **cfg, activation=nn.Sigmoid())
	out4, _mask_ = b4(x, mask)
	assert_block_contains(b4, ['PartialConv2d', 'Sigmoid'])
	assert_block_not_contains(b4, ['BatchNorm2d', 'ReLU', 'Upsample'])
	assert_size_match(out4.size(), [1, 4, 32, 32]) and assert_size_match(_mask_.size(), [1, 1, 32, 32])

	# with upsample
	cfg = {'in_channels': 3, 'out_channels': 4, 'kernel_size': 3, 'stride': 1, 'padding': 1}
	# conv-bn-relu
	print('upsample + conv-bn-relu')
	b1 = block(True, False, False, **cfg)
	out1 = b1(x)
	assert_block_contains(b1, ['Conv2d', 'BatchNorm2d', 'ReLU', 'Upsample'])
	assert_size_match(out1.size(), [1, 4, 128, 128])

	# conv-relu
	print('upsample + conv-relu')
	b2 = block(True, False, True, **cfg)
	out2 = b2(x)
	assert_block_contains(b2, ['Conv2d', 'ReLU', 'Upsample'])
	assert_block_not_contains(b2, ['BatchNorm2d'])
	assert_size_match(out2.size(), [1, 4, 128, 128])

	# pconv-in-lrelu
	print('upsample + pconv-in-lrelu')
	b3 = block(True, True, False, **cfg, norm=nn.InstanceNorm2d, activation=nn.LeakyReLU(0.2))
	out3, _mask_ = b3(x, mask2)
	assert_block_contains(b3, ['PartialConv2d', 'InstanceNorm2d', 'LeakyReLU', 'Upsample'])
	assert_block_not_contains(b3, ['BatchNorm2d', 'ReLU'])
	assert_size_match(out3.size(), [1, 4, 128, 128]) and assert_size_match(_mask_.size(), [1, 1, 128, 128])

	# pconv-sigmoid
	print('upsample + pconv-sigmoid')
	b4 = block(True, True, True, **cfg, activation=nn.Sigmoid())
	out4, _mask_ = b4(x, mask2)
	assert_block_contains(b4, ['PartialConv2d', 'Sigmoid', 'Upsample'])
	assert_block_not_contains(b4, ['BatchNorm2d', 'ReLU'])
	assert_size_match(out4.size(), [1, 4, 128, 128]) and assert_size_match(_mask_.size(), [1, 1, 128, 128])


def test_res_block():
	block = ResBlock
	x = Variable(torch.randn(1, 3, 64, 64))
	mask = Variable(torch.bernoulli(torch.rand(1, 1, 64, 64)))
	mask2 = Variable(torch.bernoulli(torch.rand(1, 1, 128, 128)))

	# without upsample
	cfg = {'in_channels': 3, 'out_channels': 4, 'kernel_size': 3, 'stride': 2, 'padding': 1}
	# resblock with conv and bn and relu
	print('resblock with conv and bn and relu')
	b1 = block(False, False, False, **cfg)
	out1 = b1(x)
	assert_block_contains(b1, ['Conv2d', 'BatchNorm2d', 'ReLU'])
	assert_block_not_contains(b1, ['Upsample'])
	assert_size_match(out1.size(), [1, 4, 32, 32])

	# resblock with pconv and in and lrelu
	print('resblock with pconv and in and lrelu')
	b2 = block(False, True, False, **cfg, norm=nn.InstanceNorm2d, activation=nn.LeakyReLU(0.2))
	out2, _mask_ = b2(x, mask)
	assert_block_contains(b2, ['PartialConv2d', 'InstanceNorm2d', 'LeakyReLU', 'ReLU'])
	assert_block_not_contains(b2, ['BatchNorm2d', 'Upsample'])
	assert_size_match(out2.size(), [1, 4, 32, 32]) and assert_size_match(_mask_.size(), [1, 1, 32, 32])

	# resblock with pconv and bn and sigmoid
	print('resblock with pconv and bn and sigmoid')
	b3 = block(False, True, True, **cfg, activation=nn.Sigmoid())
	out3, _mask_ = b3(x, mask)
	assert_block_contains(b3, ['PartialConv2d', 'Sigmoid', 'ReLU', 'BatchNorm2d'])
	assert_block_not_contains(b3, ['Upsample'])
	assert_size_match(out3.size(), [1, 4, 32, 32]) and assert_size_match(_mask_.size(), [1, 1, 32, 32])

	# with upsample
	cfg = {'in_channels': 3, 'out_channels': 4, 'kernel_size': 3, 'stride': 1, 'padding': 1}
	# resblock with conv and bn and relu
	print('upsample + resblock with conv and bn and relu')
	b1 = block(True, False, False, **cfg)
	out1 = b1(x)
	assert_block_contains(b1, ['Conv2d', 'BatchNorm2d', 'ReLU', 'Upsample'])
	assert_size_match(out1.size(), [1, 4, 128, 128])

	# resblock with pconv and in and lrelu
	print('upsample + resblock with pconv and in and lrelu')
	b2 = block(True, True, False, **cfg, norm=nn.InstanceNorm2d, activation=nn.LeakyReLU(0.2))
	out2, _mask_ = b2(x, mask2)
	assert_block_contains(b2, ['PartialConv2d', 'InstanceNorm2d', 'LeakyReLU', 'ReLU', 'Upsample'])
	assert_block_not_contains(b2, ['BatchNorm2d'])
	assert_size_match(out2.size(), [1, 4, 128, 128]) and assert_size_match(_mask_.size(), [1, 1, 128, 128])

	# resblock with pconv and bn and sigmoid
	print('upsample + resblock with pconv and bn and sigmoid')
	b3 = block(True, True, True, **cfg, activation=nn.Sigmoid())
	out3, _mask_ = b3(x, mask2)
	assert_block_contains(b3, ['PartialConv2d', 'Sigmoid', 'ReLU', 'BatchNorm2d', 'Upsample'])
	assert_size_match(out3.size(), [1, 4, 128, 128]) and assert_size_match(_mask_.size(), [1, 1, 128, 128])


@pytest.mark.parametrize('block,k', [(b, _k) for b in ['basic', 'residual'] for _k in range(8)] + \
					[(b, _k+random.random()) for b in ['basic', 'residual'] for _k in range(7)])
def test_create_pg_generator(block, k):
	k_int = 7-int(k)
	x = Variable(torch.randn(1, 3, 2**(k_int+2), 2**(k_int+2)))
	mask = Variable(torch.bernoulli(torch.rand(1, 1, 2**(k_int+2), 2**(k_int+2))))

	# only conv
	print('only conv')
	config = {'g_input': 'masked_X', 'decoder_partial_conv': False, 'init_type': 'kaiming', \
			'learning_residual': False, 'block': block, 'progressive_growing': True, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, False, False, False, 'kaiming')
	out1 = g(x, phase=k)
	assert out1.size() == x.size()

	# only partial conv on encoder
	print('only partial conv on encoder')
	config = {'g_input': 'masked_X+mask', 'decoder_partial_conv': False, 'init_type': 'kaiming', \
			'learning_residual': False, 'block': block, 'progressive_growing': True, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, False, True, False, 'kaiming')
	out2 = g(x, mask, phase=k)
	assert out2.size() == x.size()

	# full partial conv
	print('full partial conv')
	config = {'g_input': 'masked_X+mask', 'decoder_partial_conv': True, 'init_type': 'kaiming', \
			'learning_residual': False, 'block': block, 'progressive_growing': True, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, False, True, True, 'kaiming')
	out3 = g(x, mask, phase=k)
	assert out3.size() == x.size()

	# conv + learning residual
	print('conv + learning residual')
	config = {'g_input': 'masked_X', 'decoder_partial_conv': False, 'init_type': 'kaiming', \
			'learning_residual': True, 'block': block, 'progressive_growing': True, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, True, False, False, 'kaiming')
	out4 = g(x, phase=k)
	assert out4.size() == x.size()

	# only partial conv on encoder + learning residual
	print('only partial conv on encoder + learning residual')
	config = {'g_input': 'masked_X+mask', 'decoder_partial_conv': False, 'init_type': 'kaiming', \
			'learning_residual': True, 'block': block, 'progressive_growing': True, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, True, True, False, 'kaiming')
	out5 = g(x, mask, phase=k)
	assert out5.size() == x.size()

	# full partial conv + learning residual
	print('full partial conv + learning residual')
	config = {'g_input': 'masked_X+mask', 'decoder_partial_conv': True, 'init_type': 'kaiming', \
			'learning_residual': True, 'block': block, 'progressive_growing': True, 'rdpcl': False}
	g = create_Generator(config)
	assert_configs(g, True, True, True, 'kaiming')
	out6 = g(x, mask, phase=k)
	assert out6.size() == x.size()

