import torch
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

K = 10
"""Kernel (filter) size. 10 is given as optimal in the CAML paper."""

D = 50
"""The number of filter maps. 50 is optimal in the CAML paper."""

class ConvolutionalAttentionPool(torch.nn.Module):
    def __init__(self, config, conditioning_layer : bool = False):
        super().__init__()

        ### HACK: use a bert model to get our embedding matrix.
        bert = RobertaModel(config)
        W    = bert.embeddings.word_embeddings.weight
        
        self._num_embeddings = W.shape[0]
        self._embedding_dim  = W.shape[1]

        self._embed = bert.embeddings.word_embeddings
        ###

        self._num_labels = config.num_labels

        # Conv layer
        self._conv       = torch.nn.Conv1d(
            in_channels  = self._embedding_dim,
            out_channels = D,
            kernel_size  = K,
            padding      = K//2
        )
        
        # Context vector and final linear layer
        self._u     = torch.nn.Linear(D, self._num_labels)
        self._final = torch.nn.Linear(D, self._num_labels)

        if conditioning_layer:
            # NOTE: This isn't part of the normal CAML model.
            # this gives |L|**2 params to optimize. If L is large, consider adding a smaller,
            # intermediate layer to reduce this computational cost.

            self._conditioning  = torch.nn.Linear(self._num_labels, self._num_labels)
            
            # XXX INITIALIZE TO IDENTITTY AND FREEZE THE CONDITIONING LAYER!
            self._conditioning.weight = torch.nn.Parameter(torch.eye(self._num_labels))
            self._conditioning.weight.requires_grad = False
            # self._conditioning2 = torch.nn.Linear(self._num_labels, self._num_labels)
            # torch.nn.init.xavier_uniform_(self._conditioning.weight)
            # torch.nn.init.xavier_uniform_(self._conditioning2.weight)
            print(f"WARNING : Training CAML with an inter-label conditioning layer with {len(list(self._conditioning.parameters()))} parameters")
        else:
            self._conditioning = None

        # Initialize the weights of each module.        
        torch.nn.init.xavier_uniform_(self._conv .weight)
        torch.nn.init.xavier_uniform_(self._u    .weight)
        torch.nn.init.xavier_uniform_(self._final.weight)

    def forward(
        self,
        input_ids : torch.Tensor = None,
        attention_mask           = None,
        token_type_ids           = None,
        position_ids             = None,
        head_mask                = None,
        inputs_embeds            = None,
        labels                   = None,
        output_attentions        = None,
        output_hidden_states     = None,
        return_dict              = None,
    ):
        # input_ids.shape = batch_size * ? * tokens
        input_ids1 = input_ids.squeeze(1)
        if not torch.all((input_ids1 >= 0) & (input_ids1 <= self._num_embeddings)):
            loss               = torch.Tensor([0.])
            loss.requires_grad = True
            return SequenceClassifierOutput(loss.sum(), logits=torch.zeros_like(input_ids1))
            raise Runtimerror("ONE OR MORE ID'S ISN'T IN THE EMBEDDINGS")

        x0 = self._embed(input_ids1)

        # dims = len(x0.shape)
        # assert dims == 3, breakpoint(); "x0 should have three dimensions here but has {} with shape {}".format(dims, x0.shape)
        # shape:  batch_size * chunk_size * seq. length * embedding dim.

        # tanh(conv(W, x0) + bias)
        x1 = self._conv(x0.transpose(1, 2))
        x2 = torch.tanh(x1.transpose(1, 2))
        
        # shapes:
        #   self._u: num_labels * number of convolutional filters
        #   x2     : batch_size * number of convolutional filters *
        # attention0 = self._u(x2.transpose(1,2))
        attention0 = self._u.weight.matmul(x2.transpose(1,2))
        attention1 = torch.softmax(attention0, dim=2)

        m = attention1 @ x2
        
        # Compute ŷ; multiply the final per-label linear layer
        # with the per-label attentions outputs (m)
        # then reduce the resulting products via summation.
        ŷ0 = self._final.weight.mul(m)
        ŷ1 = ŷ0.sum(dim=2)
        ŷ2 = ŷ1.add(self._final.bias)
        # batch size * label size

        # Apply inter-label conditioning if a conditioning layer exists.
        # Simply alias ŷ2 to ŷ3 if not.
        if self._conditioning:
            ŷ3 = self._conditioning(ŷ2)
            # ŷ3prime = torch.relu(self._conditioning(ŷ2))
            # ŷ3      = self._conditioning(ŷ3prime)
        else:
            ŷ3 = ŷ2

        loss = torch.binary_cross_entropy_with_logits(ŷ3, labels)
        return SequenceClassifierOutput(
            loss       = loss.sum(),
            logits     = ŷ2,
            attentions = attention1,
        )
