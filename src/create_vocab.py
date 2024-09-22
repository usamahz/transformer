import sentencepiece as spm
import os

def train_spm(input_file, model_prefix, vocab_size):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        unk_id=0,
        pad_id=1,
        bos_id=2,
        eos_id=3
    )

# Paths to your training data files
english_file = 'data/english_sentences.txt'
french_file = 'data/french_sentences.txt'

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Create sample data if it doesn't exist
if not os.path.exists(english_file) or not os.path.exists(french_file):
    print("Creating sample data...")
    with open(english_file, 'w') as f:
        f.write("""Hello, how are you?
I love machine learning.
The weather is beautiful today.
Can you help me with this translation?
What time is it?
I'm hungry.
Where is the nearest restaurant?
Thank you very much.
Have a nice day!
Goodbye!
The cat is sleeping on the couch.
I enjoy reading books in my free time.
She plays the piano every evening.
We're going to the beach this weekend.
He's studying computer science at university.""")

    with open(french_file, 'w') as f:
        f.write("""Bonjour, comment allez-vous ?
J'adore l'apprentissage automatique.
Le temps est magnifique aujourd'hui.
Pouvez-vous m'aider avec cette traduction ?
Quelle heure est-il ?
J'ai faim.
Où est le restaurant le plus proche ?
Merci beaucoup.
Bonne journée !
Au revoir !
Le chat dort sur le canapé.
J'aime lire des livres pendant mon temps libre.
Elle joue du piano tous les soirs.
Nous allons à la plage ce week-end.
Il étudie l'informatique à l'université.""")

# Train English tokenizer
train_spm(english_file, 'vocab/english', 150)  # Adjusted vocab size

# Train French tokenizer
train_spm(french_file, 'vocab/french', 150)  # Adjusted vocab size

print("Vocabulary files created successfully!")
