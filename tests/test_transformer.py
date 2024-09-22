import torch
import unittest
from src.model.transformer import Transformer

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.src_seq_length = 10
        self.tgt_seq_length = 12
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1000
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.num_layers = 6
        self.dropout = 0.1
        self.transformer = Transformer(self.src_vocab_size, self.tgt_vocab_size, self.d_model, self.num_heads, self.d_ff, self.num_layers, self.dropout)

    def test_transformer_output_shape(self):
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_seq_length))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_seq_length))
        src_mask = torch.ones(self.batch_size, 1, 1, self.src_seq_length).bool()
        tgt_mask = torch.tril(torch.ones(self.tgt_seq_length, self.tgt_seq_length)).unsqueeze(0).unsqueeze(0)

        output = self.transformer(src, tgt, src_mask, tgt_mask)
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_length, self.tgt_vocab_size))

    def test_transformer_mask_application(self):
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_seq_length))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_seq_length))
        src_mask = torch.ones(self.batch_size, 1, 1, self.src_seq_length).bool()
        src_mask[:, :, :, 5:] = False  # Mask out the second half of the source sequence
        tgt_mask = torch.tril(torch.ones(self.tgt_seq_length, self.tgt_seq_length)).unsqueeze(0).unsqueeze(0)
        tgt_mask[:, :, 6:, 6:] = 0  # Mask out the second half of the target sequence

        output = self.transformer(src, tgt, src_mask, tgt_mask)
        
        unmasked_mean = output[:, :6].abs().mean()
        masked_mean = output[:, 6:].abs().mean()
        
        print(f"Unmasked mean: {unmasked_mean.item():.4f}")
        print(f"Masked mean: {masked_mean.item():.4f}")
        
        self.assertLess(masked_mean, unmasked_mean * 0.1, 
                        "Masked output should be significantly lower than unmasked output")

    def test_transformer_inference(self):
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_seq_length))
        src_mask = torch.ones(self.batch_size, 1, 1, self.src_seq_length).bool()

        max_length = 20
        start_symbol = 0

        output = self.transformer.inference(src, src_mask, max_length, start_symbol)
        self.assertEqual(output.shape, (self.batch_size, max_length))
        self.assertTrue((output >= 0).all() and (output < self.tgt_vocab_size).all())

if __name__ == '__main__':
    unittest.main()
