import torch
import torch.nn as nn
from model.transformer import Transformer
from utils.data_processing import create_src_mask
import sentencepiece as spm

class Translator:
    def __init__(self, model_path, src_vocab_path, tgt_vocab_path, device):
        self.device = device
        
        # Load vocabularies
        self.src_sp = spm.SentencePieceProcessor()
        self.tgt_sp = spm.SentencePieceProcessor()
        self.src_sp.Load(src_vocab_path)
        self.tgt_sp.Load(tgt_vocab_path)
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path):
        # Initialize model (make sure these parameters match your trained model)
        model = Transformer(
            src_vocab=150,  # Change src_vocab_size to src_vocab
            tgt_vocab=150,  # Change tgt_vocab_size to tgt_vocab
            d_model=512,
            num_heads=8,
            num_layers=6,  # Add this parameter
            d_ff=2048,
            dropout=0.1,
            max_seq_length=100  # Add this parameter, adjust the value as needed
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def translate(self, src_sentence, max_len=50):
        self.model.eval()
        src_tokens = self.src_sp.EncodeAsIds(src_sentence)
        src = torch.LongTensor(src_tokens).unsqueeze(0).to(self.device)
        src_mask = create_src_mask(src)
        
        ys = self.model.generate(src, src_mask, max_len)
        
        tgt_tokens = ys.squeeze().tolist()
        return self.tgt_sp.DecodeIds(tgt_tokens)

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = Translator(
        model_path="saved_models/final_model.pth",
        src_vocab_path="vocab/english.model",
        tgt_vocab_path="vocab/french.model",
        device=device
    )
    
    # Example sentences
    english_sentences = [
        "Hello, how are you?",
        "I love machine learning.",
        "The weather is beautiful today.",
        "Can you help me with this translation?",
    ]
    
    for sentence in english_sentences:
        translation = translator.translate(sentence)
        print(f"English: {sentence}")
        print(f"French: {translation}")
        print()
