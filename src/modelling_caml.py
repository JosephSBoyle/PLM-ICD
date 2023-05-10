import torch
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

K = 10
"""Kernel (filter) size. 10 is given as optimal in the CAML paper."""

D = 50
"""The number of filter maps. 50 is optimal in the CAML paper."""

class ConvolutionalAttentionPool(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        ### HACK: use a bert model to get our embedding matrix.
        # bert = BertModel(config)
        bert = RobertaModel(config)
        W = bert.embeddings.word_embeddings.weight
        
        self._num_embeddings = W.shape[0]
        self._embedding_dim  = W.shape[1]

        # self._embed = torch.nn.Embedding(self._num_embeddings, self._embedding_dim, padding_idx=0)
        self._embed = bert.embeddings.word_embeddings
        ###

        self._num_labels = config.num_labels

        # Conv layer
        self._conv = torch.nn.Conv1d(
            in_channels  = self._embedding_dim,
            out_channels = D,
            kernel_size  = K,
            padding      = K//2
        )
        
        # Context vector and final linear layer
        self._u     = torch.nn.Linear(D, self._num_labels)
        self._final = torch.nn.Linear(D, self._num_labels)

        # Initialize the weights of each module.        
        torch.nn.init.xavier_uniform_(self._conv .weight)
        torch.nn.init.xavier_uniform_(self._u    .weight)
        torch.nn.init.xavier_uniform_(self._final.weight)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        input_ids1 = input_ids.squeeze()
        if not torch.all((input_ids1 >= 0) & (input_ids1 <= self._num_embeddings)):
            loss = torch.Tensor([0.])
            loss.requires_grad = True
            return SequenceClassifierOutput(loss.sum(), logits=torch.zeros_like(input_ids1))
            raise Runtimerror("ONE OR MORE ID'S ISN'T IN THE EMBEDDINGS")

        x0 = self._embed(input_ids1)

        dims = len(x0.shape)
        assert dims == 3, breakpoint(); "x0 should have three dimensions here but has {} with shape {}".format(dims, x0.shape)
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
        ### Now compute the BCE loss

        loss = torch.binary_cross_entropy_with_logits(ŷ2, labels)
        return SequenceClassifierOutput(
            loss       = loss.sum(),
            logits     = ŷ2,
            attentions = attention1,
        )
