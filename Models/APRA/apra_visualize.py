import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import Models.APRA.apra as apra
# computed variables that don't change between calls
_stacks = None
_slices = None
_widths = None


def plot_tensors(fig, tensors):
	global _stacks
	global _slices
	global _widths
	# compute the layout
	# color grading
	maxabs = max(
		0.01,
		*(torch.max(torch.abs(t)).item() for t in tensors)
	)
	# compute stacking
	hw_ratio = [t.shape[0]/t.shape[1] for t in tensors]

	if _stacks is None:
		stack_t = 8
		_stacks = []
		curhw = 0
		last_h = 0
		for i, hw in enumerate(hw_ratio):
			# only stack tensors in layout if dims are compatible and if there's space
			if curhw and curhw + hw < stack_t and last_h == tensors[i].shape[0]:
				_stacks[-1].append(i)
			else:
				# t is first tensor of column
				curhw = 0
				_stacks.append([i])
			curhw += hw
			last_h = tensors[i].shape[1]

	# compute slices
	rows = max(len(col) for col in _stacks)

	if _slices is None:
		_slices = []
		for col in _stacks:
			# normalize each hw to #rows
			colhw = [hw_ratio[i] for i in col]
			colhw = [hw * rows / sum(colhw) for hw in colhw]
			s = 0
			start = 0
			for hw in colhw:
				s += hw
				end = min(max(start + 1, int(s)), rows)
				_slices.append(slice(start, end))
				start = end

	# compute width
	if _widths is None:
		maxwidths = [max(tensors[i].shape[1] for i in col) for col in _stacks]
		total = sum(maxwidths)
		_widths = [max(mw/total, 0.06) for mw in maxwidths] + [.03]
	# setup layout
	cols = len(_stacks) + 1
	gs = fig.add_gridspec(rows, cols, width_ratios=_widths)
	# plot tensors
	axes = []
	im = None
	for c, col in enumerate(_stacks):
		for i in col:
			axes.append(fig.add_subplot(gs[_slices[i], c]))
			im = axes[-1].matshow(tensors[i], vmin=-maxabs, vmax=maxabs, aspect="auto")
	# plot colorbar and title
	ax = fig.add_subplot(gs[:, -1])
	fig.colorbar(im, cax=ax)
	gs.tight_layout(fig, rect=[0.01, 0, 1, 0.97])
	return axes


def view_model(fig, module: "apra.APRAModule", it: int, grad: bool):
	# prepare the data
	l = lambda w: w.grad.cpu() if grad else w.detach().cpu()
	cembed = l(module.embed.embedding.weight)
	cconv1 = l(module.embed.convs[0].weight)[0]
	cconv2 = l(module.embed.convs[3].weight)[0]
	cconv3 = l(module.embed.convs[6].weight)[0]
	cconv4 = l(module.embed.convs[9].weight)[0]
	cf = l(module.embed.ff.weight)
	mf1 = l(module.ff[0].weight).transpose(0, 1)
	mf2 = l(module.ff[2].weight).transpose(0, 1)
	tensors = (cembed, cconv1, cconv2, cconv3, cconv4, cf, mf1, mf2)
	# plot the tensors
	axes = plot_tensors(fig, tensors)
	# adjust some labels
	axes[0].set_yticks(list(range(len(module.embed.cvoc))))
	axes[0].set_yticklabels(list(module.embed.cvoc.keys()))
	# title the figure
	fig.suptitle(f"ARPA {'Gradients' if grad else 'Weights'} at it={it}")
