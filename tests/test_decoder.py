import torch
import unittest
from src.model.decoder import DecoderLayer, Decoder

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_length = 10
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.num_layers = 6
        self.dropout = 0.1
        self.decoder_layer = DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        self.decoder = Decoder(self.d_model, self.num_heads, self.d_ff, self.num_layers, self.dropout)

    def test_decoder_layer_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_length, self.d_model)
        encoder_output = torch.randn(self.batch_size, self.seq_length, self.d_model)
        src_mask = torch.ones(self.batch_size, 1, 1, self.seq_length).bool()
        tgt_mask = torch.tril(torch.ones(self.seq_length, self.seq_length)).unsqueeze(0).unsqueeze(0)

        output = self.decoder_layer(x, encoder_output, src_mask, tgt_mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))

    def test_decoder_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_length, self.d_model)
        encoder_output = torch.randn(self.batch_size, self.seq_length, self.d_model)
        src_mask = torch.ones(self.batch_size, 1, 1, self.seq_length).bool()
        tgt_mask = torch.tril(torch.ones(self.seq_length, self.seq_length)).unsqueeze(0).unsqueeze(0)

        output = self.decoder(x, encoder_output, src_mask, tgt_mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))

    def test_decoder_mask_application(self):
        x = torch.randn(self.batch_size, self.seq_length, self.d_model)
        encoder_output = torch.randn(self.batch_size, self.seq_length, self.d_model)
        src_mask = torch.ones(self.batch_size, 1, 1, self.seq_length).bool()
        tgt_mask = torch.tril(torch.ones(self.seq_length, self.seq_length)).unsqueeze(0).unsqueeze(0)
        tgt_mask[:, :, 5:, 5:] = 0  # Mask out the second half of the sequence

        output = self.decoder(x, encoder_output, src_mask, tgt_mask)
        self.assertFalse(torch.allclose(output[:, :5], torch.zeros_like(output[:, :5])))
        self.assertTrue(torch.allclose(output[:, 5:], torch.zeros_like(output[:, 5:])))

if __name__ == '__main__':
    unittest.main()
