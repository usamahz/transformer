import torch
import unittest
from src.model.attention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_length = 10
        self.d_model = 512
        self.num_heads = 8
        self.d_k = self.d_model // self.num_heads
        self.attention = MultiHeadAttention(self.d_model, self.num_heads)

    def test_output_shape(self):
        query = torch.randn(self.batch_size, self.seq_length, self.d_model)
        key = torch.randn(self.batch_size, self.seq_length, self.d_model)
        value = torch.randn(self.batch_size, self.seq_length, self.d_model)
        mask = torch.ones(self.batch_size, 1, 1, self.seq_length).bool()

        output = self.attention(query, key, value, mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))

    def test_mask_application(self):
        query = torch.randn(self.batch_size, self.seq_length, self.d_model)
        key = torch.randn(self.batch_size, self.seq_length, self.d_model)
        value = torch.randn(self.batch_size, self.seq_length, self.d_model)
        mask = torch.ones(self.batch_size, 1, 1, self.seq_length).bool()
        mask[:, :, :, 5:] = False  # Mask out the second half of the sequence

        output = self.attention(query, key, value, mask)
        
        # Check if the masked part has significantly lower values
        unmasked_mean = output[:, :5].abs().mean()
        masked_mean = output[:, 5:].abs().mean()
        
        print(f"Unmasked mean: {unmasked_mean.item():.4f}")
        print(f"Masked mean: {masked_mean.item():.4f}")
        
        self.assertLess(masked_mean, unmasked_mean * 0.1, 
                        "Masked output should be significantly lower than unmasked output")

    def test_attention_weights(self):
        query = torch.randn(self.batch_size, self.seq_length, self.d_model)
        key = torch.randn(self.batch_size, self.seq_length, self.d_model)
        value = torch.randn(self.batch_size, self.seq_length, self.d_model)
        mask = None

        output, attention_weights = self.attention(query, key, value, mask, return_attention=True)
        self.assertEqual(attention_weights.shape, (self.batch_size, self.num_heads, self.seq_length, self.seq_length))
        self.assertTrue(torch.allclose(attention_weights.sum(dim=-1), torch.ones(self.batch_size, self.num_heads, self.seq_length)))

if __name__ == '__main__':
    unittest.main()
