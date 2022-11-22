#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# copied from
# https://github.com/facebookresearch/PyTorch-BigGraph/blob/master/torchbiggraph/tensorlist.py

import torch


def _extract_intervals(offsets, sizes, data):
    """Select contiguous intervals of rows, given their offsets and sizes.
    E.g. suppose offsets = [35, 70, 90], sizes = [3, 2, 4], then this will
    return
            (torch.LongTensor(0, 3, 5),
            data[torch.LongTensor([35, 36, 37, 70, 71, 90, 91, 92, 93])])
    """
    offsets = offsets.long()
    sizes = sizes.long()
    res_rows = sizes.sum().item()
    assert offsets.size(0) == sizes.size(0)

    non_zero_size = sizes != 0
    if non_zero_size.long().sum() == 0:
        return torch.zeros(offsets.size(0) + 1).long(), data.new()

    new_offsets = torch.cat([torch.LongTensor([0]), sizes.cumsum(0)])
    sizes_nz = sizes[non_zero_size]
    offsets_nz = offsets[non_zero_size]

    res_delta = torch.LongTensor(res_rows).fill_(1)
    res_delta[0] = offsets_nz[0]

    if offsets_nz.size(0) > 1:
        input_delta = offsets_nz[1:] - offsets_nz[:-1] - sizes_nz[:-1]  # [32, 18]
        res_row_offsets = sizes_nz.cumsum(0)[:-1]  # [3, 5]
        res_delta[res_row_offsets] += input_delta  # [35, 1, 1, 33, 1, 19, 1, 1, 1]

    res_offsets = res_delta.cumsum(0)  # [35, 36, 37, 70, 71, 90, 91, 92, 93]
    res = data[res_offsets]

    return new_offsets, res


class TensorList(object):
    """A list of tensors of different sizes, backed by a (offset, size, data)
    tuple.
    Indexing by LongTensor returns a new TensorList with the selected list
    elements (similar to indexing a torch index_select_).
    Indexing by an int returns a torch.Tensor with that list element.
    """

    @classmethod
    def cat(cls, elements):
        offsets, data = zip(*[[x.offsets, x.data] for x in elements])
        offsets = list(offsets)
        batch_offset = torch.LongTensor([o[-1] for o in offsets]).cumsum(0)
        for j in range(len(offsets) - 1):
            offsets[j + 1] = offsets[j + 1][1:] + batch_offset[j]
        return cls(torch.cat(offsets), torch.cat(data))

    @classmethod
    def empty(cls, num_tensors=0):
        return cls(
            torch.zeros((), dtype=torch.long).expand((num_tensors + 1,)),
            torch.empty((0,), dtype=torch.long),
        )

    def new(self):
        return type(self)(self.offsets.new_zeros((1,)), self.data.new_empty((0,)))

    def __init__(self, offsets, data):
        # some sanity checks
        assert isinstance(offsets, torch.LongTensor)
        assert offsets.ndimension() == 1
        assert offsets[0] == 0
        assert offsets[-1] == (data.size(0) if data.ndimension() > 0 else 0)

        # Temporary workaround for below PyTorch bug
        # https://github.com/pytorch/pytorch/issues/5719
        if data.numel() == 0 and data.storage().size() == 0:
            storage = data.storage()
            storage.resize_(storage.size() + 1)

        self.offsets = offsets
        self.data = data

    def __getitem__(self, index):
        if isinstance(index, torch.LongTensor):
            offsets_sub = self.offsets[index]
            sizes_sub = self.offsets[index + 1] - offsets_sub
            new_offsets, new_data = _extract_intervals(offsets_sub, sizes_sub, self.data)

            return TensorList(new_offsets, new_data)
        elif isinstance(index, int):
            if self.offsets[index] != self.offsets[index + 1]:
                return self.data[self.offsets[index] : self.offsets[index + 1]]
            else:
                return self.data.new()
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("Expected slice with step 1, got %d" % step)
            new_offsets = self.offsets[start : stop + 1]
            new_data = self.data[new_offsets[0] : new_offsets[-1]]
            new_offsets = new_offsets - new_offsets[0]
            return TensorList(new_offsets, new_data)
        else:
            raise KeyError("Unknown index type: %s" % type(index))

    def __eq__(self, other):
        if not isinstance(other, TensorList):
            return NotImplemented
        return torch.equal(self.offsets, other.offsets) and torch.equal(self.data, other.data)

    def __len__(self):
        return self.offsets.size(0) - 1

    def __iadd__(self, other):
        if isinstance(other, int):
            self.data += other
            return self
        else:
            raise NotImplementedError()

    def __isub__(self, other):
        if isinstance(other, int):
            self.data -= other
            return self
        else:
            raise NotImplementedError()

    def size(self, dim=None):
        # to have similar appearance with other tensor types
        assert dim == 0 or dim is None, "TensorList can only have 1 dimension"
        if dim is None:
            return torch.Size([len(self)])
        else:
            return len(self)

    def nelement(self):
        return self.data.nelement()

    def clone(self):
        return self.__class__(self.offsets, self.data.clone())

    def __repr__(self):
        if self.offsets.nelement() < 100 or self.data.nelement() < 1000:
            return "TensorList( [%s] )" % " , ".join(
                str(self[i].tolist()) for i in range(len(self))
            )
        return "TensorList{offsets=%s, data=%s}" % (self.offsets, self.data)

    def apply(self, F):
        return self.__class__(self.offsets, F(self.data))

    def combine(self, other, F):
        if isinstance(other, TensorList):
            assert torch.equal(self.offsets, other.offsets)
            assert self.data.shape[0] == other.data.shape[0]
            res = self.__class__(self.offsets, F(self.data, other.data))
        else:
            res = self.__class__(self.offsets, F(self.data, other))
        assert res.data.shape[0] == self.data.shape[0]
        return res

    def lengths(self):
        return self.offsets[1:] - self.offsets[:-1]

    def unsqueeze(self, dim):
        return self.apply(lambda x: x.unsqueeze(dim))

    def view(self, *args):
        return self.apply(lambda x: x.view(*args))

    def __add__(self, other):
        return self.combine(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self.combine(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self.combine(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self.combine(other, lambda x, y: x / y)

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return self.data.sum()
        # We're only going to agree to sum across "inner dimensions"
        if dim < 0:
            dim = self.data.ndimension() + dim
        assert dim > 0, "Can't sum along the 'list' dimension"
        return self.__class__(self.offsets, self.data.sum(dim, keepdim=keepdim))

    def to_padded(self, padding_value=0, total_length=None):
        """
        Convert to a B x D padded sequence.

        We'll be lazy for now and assume 1D elements, because that's all we use.

        The way we do this is a bit tricky. Basically, we solve for a 1D tensor
        that codes for the offsets of each element of self.data in the padded
        tensor, and then we do an index copy of data into the padded tensor.

        This only works when all rows are non-empty, so we have to first remove
        the empty rows.
        """

        N = len(self)
        lengths = self.lengths()
        if total_length is None:
            total_length = int(lengths.max())

        res = self.data.new_empty(N, total_length).fill_(padding_value)

        nz_rows = lengths.nonzero(as_tuple=False)[:, 0]
        if len(nz_rows) != len(lengths):
            # operate on the non-empty rows, and insert back in
            res_nz = self[nz_rows].to_padded(padding_value, total_length)
            res[nz_rows] = res_nz
        else:
            insertion_delta = torch.ones(self.data.shape, dtype=torch.long)
            insertion_delta[self.offsets[1:-1]] += total_length - lengths[:-1]
            insertion_delta[0] = 0
            insertion_idx = insertion_delta.cumsum(0)
            res.view(-1)[insertion_idx] = self.data

        return res

    @classmethod
    def from_padded(cls, data, padding_value=0):
        """
        Convert a B x D padded sequence to TensorList.

        We'll be lazy for now and assume 1D elements, because that's all we use.
        """
        assert data.ndimension() == 2
        valid_mask = data != padding_value
        lengths = valid_mask.sum(dim=1)
        offsets = lengths.cumsum(0)
        offsets = torch.cat((offsets.new_zeros(1), offsets), dim=0)
        packed = data.view(-1)[valid_mask.view(-1)]

        return cls(offsets, packed)


if __name__ == "__main__":
    tl = TensorList(
        torch.tensor([0, 0, 3, 9, 9, 11], dtype=torch.long),
        torch.tensor([1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2]),
    )
    print(tl.to_padded())
