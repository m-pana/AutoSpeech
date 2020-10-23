- the loss for the architecture step is computed in model_search.Network._loss
- entropy is computed in functions.train (line 52)

checkpoint files contain the following
- 'epoch': current training
- 'state_dict': the model's state dict (i.e. weights of learnable params
- 'best_acc1': best_acc1
- 'optimizer': optimizer's
- 'arch': architecture params alpha (returned by call model.arch_parameters())
- 'genotype': genotype of the model. Tells what operations have been selected by the search
- 'path_helper': i have no idea

visualize.py cannot be called directly. instead, import its ``plot`` function and call

``plot(gen.normal, filename_norm)``
and
``plot(gen.reduce, filename_reduce)``

where ``gen`` is the genotype retrieved from a checkpoint file. This will save the normal and reduced cells to ``filename_norm.pdf`` and ``filename_reduce.pdf`` (yeah, pdf for some reason. I'm as puzzled as you are, my friend)

The "drop_path" function (found in utils, line 170) zeroes out values with some probability using a binary mask. It seems kinda like dropout to me, but i may be an idiot. The function itself is applied in the forward of the Cell class, and only during training (like dropout would).
