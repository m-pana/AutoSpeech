import math
import torch.nn.functional as F
from operations import *
from utils import Genotype
from utils import gumbel_softmax, drop_path


class MixedOp(nn.Module):

    def __init__(self, C, stride, PRIMITIVES):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """
        This is a forward function.
        :param x: Feature map
        :param weights: A tensor of weight controlling the path flow
        :return: A weighted sum of several path
        """
        output = 0
        for op_idx, op in enumerate(self._ops):
            if weights[op_idx].item() != 0:
                if math.isnan(weights[op_idx]):
                    raise OverflowError(f'weight: {weights}')
            output += weights[op_idx] * op(x)
        return output


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.primitives = self.PRIMITIVES['primitives_reduct' if reduction else 'primitives_normal']

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        edge_index = 0

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.primitives[edge_index])
                self._ops.append(op)
                edge_index += 1

    def forward(self, s0, s1, weights, drop_prob=0.0):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            if drop_prob > 0. and self.training:
                s = sum(
                    drop_path(self._ops[offset + j](h, weights[offset + j]), drop_prob) for j, h in enumerate(states))
            else:
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, primitives,
                 steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0.0):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.drop_path_prob = drop_path_prob

        nn.Module.PRIMITIVES = primitives

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(C_prev, self._num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._embed_dim, self._layers, self._criterion,
                            self.PRIMITIVES, drop_path_prob=self.drop_path_prob).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, discrete=False):
        # I hope this does not break anything, I don't know why it's here
        # Second take
        if len(input.shape) < 4:
            input = input.unsqueeze(1)
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if discrete:
                    weights = self.alphas_reduce
                else:
                    weights = gumbel_softmax(F.log_softmax(self.alphas_reduce, dim=-1))
            else:
                if discrete:
                    weights = self.alphas_normal
                else:
                    weights = gumbel_softmax(F.log_softmax(self.alphas_normal, dim=-1))
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
        v = self.global_pooling(s1)
        v = v.view(v.size(0), -1)
        if not self.training:
            return v

        y = self.classifier(v)

        return y

    def forward_classifier(self, v):
        y = self.classifier(v)
        return y

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self.PRIMITIVES['primitives_normal'][0])

        alph_nor = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        alph_res = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        print(f"Initializing alphas normal of shape {alph_nor.shape}")
        print(f"Initializing alphas res of shape {alph_res.shape}")
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def compute_arch_entropy(self, dim=-1):
        alpha = self.arch_parameters()[0]
        prob = F.softmax(alpha, dim=dim)
        log_prob = F.log_softmax(alpha, dim=dim)
        entropy = - (log_prob * prob).sum(-1, keepdim=False)
        print(f"Computed architecture entropy of {entropy}")
        return entropy

    def genotype(self):
        def _parse(weights, normal=True):
            PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

            gene = []
            n = 2 # number of ops entering each block here
            start = 0
            for i in range(self._steps):
                # _steps = n.sub-blocks?
                end = start + n
                W = weights[start:end].copy() # W is alphas matrix (14, 8) shape
                try:
                    # this thing ahead is unreadable.
                    # sort indices of entering edges (they go up to i+2 since each block has two more entering edges than its index.
                    # e.g. intermediate block 0 has 2 inputs (even tho intermediate block
                    # 0 would be x2...)
                    # sort them according to the maximum weight, unless there is a none.
                    # Skip the nones.
                    edges = sorted(range(i + 2), key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2] # i assume this 2 is the hardcoded number of entering edges
                        # why they did not put n here is beyond me
                except ValueError:  # This error happens when the 'none' op is not present in the ops
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2] #same as above, without the none piece
                for j in edges: # edge has 2 elements: two indices, i think
                    k_best = None # index of best operation (between two)
                    for k in range(len(W[j])):
                        if 'none' in PRIMITIVES[j]:
                            if k != PRIMITIVES[j].index('none'): # again, they never select none
                            # in conclusion, 'none' cannot be selected as operation. ever.
                            # or so it seems
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                        else:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    # Select the best operation for the edge j and add it to the genotype
                    # This means that operations are put IN ORDER in a Genotype
                    # Two ops for first block are 1st and 2nd in the list, for second block they are 3rd and 4th, and so on...
                    # the second argument of the tuple (j) is which edge the operation belongs to
                    gene.append((PRIMITIVES[start+j][k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        print(f"Produced Genotype {genotype}")
        return genotype
