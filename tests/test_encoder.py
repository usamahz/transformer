import torch
import unittest
from src.model.encoder import EncoderLayer, Encoder

class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_length = 10
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.num_layers = 6
        self.dropout = 0.1
        self.encoder_layer = EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        self.encoder = Encoder(self.d_model, self.num_heads, self.d_ff, self.num_layers, self.dropout)

    def test_encoder_layer_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_length, self.d_model)
        mask = torch.ones(self.batch_size, 1, 1, self.seq_length).bool()

        output = self.encoder_layer(x, mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))

    def test_encoder_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_length, self.d_model)
        mask = torch.ones(self.batch_size, 1, 1, self.seq_length).bool()

        output = self.encoder(x, mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))

    def test_encoder_mask_application(self):
        x = torch.randn(self.batch_size, self.seq_length, self.d_model)
        mask = torch.zeros(self.batch_size, 1, 1, self.seq_length).bool()
        mask[:, :, :, 5:] = True  # Mask out the second half of the sequence

        output = self.encoder(x, mask)
        self.assertFalse(torch.allclose(output[:, :5], torch.zeros_like(output[:, :5])))
        self.assertTrue(torch.allclose(output[:, 5:], torch.zeros_like(output[:, 5:])))

if __name__ == '__main__':
    unittest.main()
