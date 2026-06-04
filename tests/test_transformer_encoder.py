"""Tests for transformer-encoder submission: masked mean pooling, collate, vocab."""

import sys
from pathlib import Path

import torch

_submission_dir = Path(__file__).resolve().parent.parent / "submissions" / "transformer-encoder"
sys.path.insert(0, str(_submission_dir))
from main import TransformerGradePredictor, build_vocab, collate_fn  # noqa: E402


class TestMaskedMeanPooling:
    """Masked mean pooling correctness and edge cases."""

    def _make_model(self, vocab_size=10, d_model=4, **overrides):
        model = TransformerGradePredictor(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=2,
            num_layers=1,
            num_classes=3,
            **overrides,
        )
        model.eval()
        return model

    def test_masked_pooling_matches_manual(self):
        model = self._make_model()
        x = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 0]])
        mask = (x == 0).long()

        h = model.embedding(x)
        h = model.pos_encoder(h)
        h = model.transformer(h)

        mask_exp = mask.unsqueeze(-1).float()
        h_masked = h * (1.0 - mask_exp)
        lengths = (1.0 - mask_exp[:, :, 0]).sum(dim=1, keepdim=True)
        pooled = h_masked.sum(dim=1) / lengths.clamp(min=1)
        expected = model.classifier(pooled)

        actual = model(x, mask)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_padding_positions_zeroed(self):
        model = self._make_model()
        x = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
        mask = (x == 0).long()
        mask_exp = mask.unsqueeze(-1).float()

        h = model.embedding(x)
        h = model.pos_encoder(h)
        h = model.transformer(h)

        h_masked = h * (1.0 - mask_exp)

        assert (h_masked[mask.bool()] == 0).all()

    def test_valid_positions_preserved(self):
        model = self._make_model()
        x = torch.tensor([[1, 0]])
        mask = (x == 0).long()
        mask_exp = mask.unsqueeze(-1).float()

        h = model.embedding(x)
        h = model.pos_encoder(h)
        h = model.transformer(h)

        h_masked = h * (1.0 - mask_exp)

        assert torch.allclose(h_masked[0, 0], h[0, 0], atol=1e-6)

    def test_length_computation(self):
        self._make_model()
        x = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0], [4, 5, 6, 0]])

        lengths = (1 - (x == 0).float()).sum(dim=1)
        assert torch.equal(lengths, torch.tensor([2.0, 1.0, 3.0]))

    def test_all_padding_clamps_min_one(self):
        model = self._make_model()
        x = torch.tensor([[0, 0, 0], [1, 2, 0]])
        mask = (x == 0).long()

        out = model(x, mask)
        assert out.shape == (2, 3)
        assert not torch.isnan(out).any()

    def test_no_padding_equivalent_to_plain_mean(self):
        model = self._make_model()
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        mask = (x == 0).long()

        out_masked = model(x, mask)
        out_plain = model(x)
        assert torch.allclose(out_masked, out_plain, atol=1e-6)

    def test_single_valid_token(self):
        model = self._make_model()
        x = torch.tensor([[1, 0, 0, 0]])
        mask = (x == 0).long()

        h = model.embedding(x)
        h = model.pos_encoder(h)
        h = model.transformer(h)

        mask_exp = mask.unsqueeze(-1).float()
        h_masked = h * (1.0 - mask_exp)
        lengths = (1.0 - mask_exp[:, :, 0]).sum(dim=1, keepdim=True)
        pooled = h_masked.sum(dim=1) / lengths.clamp(min=1)
        expected = model.classifier(pooled)

        actual = model(x, mask)
        assert torch.allclose(actual, expected, atol=1e-6)


class TestCollateFn:
    """collate_fn returns 3-tuple (seqs, labels, mask)."""

    def test_returns_3_tuple(self):
        batch = [(torch.tensor([1, 2, 0]), 0), (torch.tensor([3, 0, 0]), 1)]
        result = collate_fn(batch)
        assert len(result) == 3

    def test_mask_is_one_where_padding(self):
        batch = [(torch.tensor([1, 2, 0]), 0), (torch.tensor([3, 0, 0]), 1)]
        _, _, mask = collate_fn(batch)
        expected = torch.tensor([[0, 0, 1], [0, 1, 1]])
        assert torch.equal(mask, expected)

    def test_mask_shape_matches_seqs(self):
        batch = [(torch.tensor([1, 2, 3, 0]), 0), (torch.tensor([4, 0, 0, 0]), 1)]
        seqs, _, mask = collate_fn(batch)
        assert mask.shape == seqs.shape

    def test_mask_is_long(self):
        batch = [(torch.tensor([1, 0]), 0)]
        _, _, mask = collate_fn(batch)
        assert mask.dtype == torch.long


class TestForwardBackwardCompat:
    """forward(x) without mask falls back to x.mean(dim=1)."""

    def test_forward_without_mask(self):
        model = TransformerGradePredictor(10, d_model=4, nhead=2, num_layers=1, num_classes=3)
        model.eval()
        x = torch.tensor([[1, 2, 3, 4]])
        out = model(x)
        assert out.shape == (1, 3)

    def test_forward_with_mask(self):
        model = TransformerGradePredictor(10, d_model=4, nhead=2, num_layers=1, num_classes=3)
        model.eval()
        x = torch.tensor([[1, 2, 3, 0]])
        mask = (x == 0).long()
        out = model(x, mask)
        assert out.shape == (1, 3)

    def test_no_padding_both_methods_equal(self):
        model = TransformerGradePredictor(10, d_model=4, nhead=2, num_layers=1, num_classes=3)
        model.eval()
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        mask = (x == 0).long()
        assert torch.allclose(model(x), model(x, mask), atol=1e-6)


class TestVocabPadToken:
    """build_vocab assigns pad token ID 0."""

    def test_pad_token_is_zero(self):
        vocab = build_vocab([["A", "B"], ["C"]])
        assert vocab["<PAD>"] == 0

    def test_other_tokens_are_positive(self):
        vocab = build_vocab([["A", "B"], ["C"]])
        for token, idx in vocab.items():
            if token != "<PAD>":
                assert idx >= 1

    def test_mask_identifies_padding(self):
        vocab = build_vocab([["A", "B", "C"]])
        token_ids = [vocab["A"], vocab["B"], vocab["C"]]
        padded = torch.tensor(token_ids + [0] * 2)
        mask = padded == 0
        assert mask[-2:].all()
        assert not mask[:3].any()
