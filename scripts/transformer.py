from constants import *
from decoder import *
from encoder import *
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)

"""
    Transformer Class
"""
class Transformer(nn.Module):
    def __init__(self, n_embed, n_head, block_size, vocab_size_x, vocab_size_y, n_layer):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.block_size = block_size

        self.token_embedding_table_x = nn.Embedding(vocab_size_x, n_embed) # paramters are num_embeddings (size of dictionary), embedding_dim (dim of embeddign vec)
        self.token_embedding_table_y = nn.Embedding(vocab_size_y, n_embed) # paramters are num_embeddings (size of dictionary), embedding_dim (dim of embeddign vec)
        self.pos_enc = PositionalEncoding(n_embed)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder_block = DecoderBlock(n_embed, n_head, block_size, n_layer)
        self.encoder_block = EncoderBlock(n_embed, n_head, block_size, n_layer)

        self.lm_head = nn.Linear(n_embed, vocab_size_y) # paramters are in_features, out_features

    """
        displays the model's attention over the source sentence for each target token generated.
        Args:
            candidate: (list) tokenized source tokens
            translation: (list) predicted target translation tokens
            attention: a tensor containing attentions scores
        Returns:
    """
    def display_attention(self, candidate, translation, attention):
        attention = attention.cpu().detach().numpy()
        # attention = [target length, source length]


        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.matshow(attention, cmap='bone')
        ax.tick_params(labelsize=15)
        ax.set_xticklabels([''] + [t for t in candidate.tolist()], rotation=45)
        ax.set_yticklabels([''] + [es_sp.DecodeIds(t) for t in translation.tolist()])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # plt.show()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'graphs/attention_graph_{timestamp}.png'
        plt.savefig(filename, bbox_inches='tight')

        plt.close()

    def forward(self, x, targets, src_mask, target_mask, c_mask):

        x_tok_emb = self.token_embedding_table_x(x)
        y_tok_emb = self.token_embedding_table_y(targets)

        x_pos_enc = self.pos_enc(x_tok_emb)
        y_pos_enc = self.pos_enc(y_tok_emb)
        
        # print("target shape: ", y_pos_enc.shape)

        enc_output = self.encoder_block(x_pos_enc, src_mask)
        dec_output, wei = self.decoder_block(y_pos_enc, enc_output, src_mask, target_mask,c_mask)
        # print("attention weight shape", wei.shape)
        # self.display_attention(x[0], targets[0], wei[0][-1])

        logits = nn.LogSoftmax(dim=-1)(self.lm_head(dec_output))
        return logits


    """
        Generate function
        tokenizes the input sentence first 
        takes a source and pos encodes and embed it 
        send it to the encoder 
        take a empty target of length of input sentence and pos encode and embed it
        send it to the decoder 
        cross attnetion between encoder and decoder outputs 
        softmax the result 

    """
    # the job of generate is to extend idx to be B by T+1, B by T+2 ....
    def generate(self, src, seq_len):
        global dropout
        dropout = 0 # turn off dropout

        self.eval() # put model in eval mode
        softmax = nn.LogSoftmax(dim=-1)

        src = torch.stack([src])
        src_mask = (src != 0).unsqueeze(1).to(device=device)  # (B, 1, L)
        src_tok_emb = self.token_embedding_table_x(src)
        src_pos_enc = self.pos_enc(src_tok_emb)

        enc_output = self.encoder_block(src_pos_enc, src_mask)

        # Define the file path where you want to save the tensor
        file_path = "tensor_data.txt"

        # Save the tensor to the text file
        with open(file_path, 'w') as file:
            for row in enc_output[0]:  # Iterate through the 96 tensors
                row_str = ' '.join(map(str, row.tolist()))  # Convert row tensor to space-separated string
                file.write(f"{row_str}\n")  # Write each row to the file

        # Confirm that the tensor has been saved
        print(f"Tensor saved to {file_path}")

        target = torch.zeros(seq_len).long().to(device)
        target[0] = 1   # set the first token to be the start token
        target = torch.stack([target])
        target_len = 0

        for i in range(1, seq_len):
            trg_mask = (target != 0).unsqueeze(1)  # (B, 1, L)
            # c_attn_mask = src_mask.expand(-1, seq_len, -1) & trg_mask.expand(-1, -1, seq_len)  # (B, L, L)

            nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
            trg_mask = (trg_mask & nopeak_mask) # (B, L, L) padding false

            c_mask = (src!=0).unsqueeze(1) * (target!=0).unsqueeze(2)  

            target_tok_emb = self.token_embedding_table_y(target)
            target_pos_enc = self.pos_enc(target_tok_emb)

            dec_output, wei = self.decoder_block(target_pos_enc, enc_output, src_mask, trg_mask, c_mask)
            output = softmax(self.lm_head(dec_output))
            output = torch.argmax(output, dim=-1) # (1, seq_len)

            last_word_id = output[0][i].item()
            # if last_word_id == 0:
                # torch.set_printoptions(threshold=100_000)
                # print(output.dtype)
                # torch.set_printoptions(profile="default") # reset

            target[0][i] = last_word_id
            target_len = i

            if last_word_id == 2:
                break
        self.display_attention(src[0], target[0], wei[0][-1])
        print(target[0])
        dropout = 0.2 # turn dropout back on
        self.train()
        return target


    def beam_search(self, src, max_len, beam_width):
        """
        Performs beam search for translation using a transformer model.
        
        Args:
            model (nn.Module): Translation Transformer model.
            src_input (Tensor): Source input sequence.
            max_len (int): Maximum length of the generated translation.
            beam_width (int): Width of the beam for search.

        Returns:
            List of tuples (translation, score).
        """
        # Ensure the model is in evaluation mode
        self.eval()

        # Encode the source input        
        src = torch.stack([src])
        src_mask = (src != 0).unsqueeze(1).to(device=device)  # (B, 1, L)
        src_tok_emb = self.token_embedding_table_x(src)
        src_pos_enc = self.pos_enc(src_tok_emb)
        with torch.no_grad():
            enc_output = self.encoder_block(src_pos_enc, src_mask)

        # Initialize the beam search
        beams = [(torch.LongTensor([1]), 0.0)]  # (current_translation, cumulative_score)
        completed_translations = []

        for _ in range(max_len):
            new_beams = []

            for translation, score in beams:
                # Get the last predicted token
                last_token = translation[-1].unsqueeze(0)

                # Decode the last token
                trg_mask = self.subsequent_mask(last_token.size(-1)).type_as(enc_output)
                # c_mask = (src!=0).unsqueeze(1) * (translation!=0).unsqueeze(2)  

                with torch.no_grad():
                    dec_output, wei = self.decoder_block(last_token, enc_output, src_mask, trg_mask, None)

                # Get the probabilities for the next token
                prob = F.softmax(dec_output, dim=-1).squeeze(0)[-1]

                # Select the top beam_width candidates based on probabilities
                topk_prob, topk_idx = prob.topk(beam_width)
                for i in range(beam_width):
                    new_translation = torch.cat((translation, topk_idx[i].unsqueeze(0)))
                    new_score = score - torch.log(topk_prob[i])
                    new_beams.append((new_translation, new_score))

            # Sort and keep the top beam_width candidates
            beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]

            # Check for completed translations (reached EOS token)
            for idx, (translation, score) in enumerate(beams):
                if translation[-1] == 2:
                    completed_translations.append((translation, score))
                    beams.pop(idx)

        # Sort the completed translations and return the top one
        completed_translations = sorted(completed_translations, key=lambda x: x[1])
        best_translation, best_score = completed_translations[0]

        return best_translation.tolist(), best_score

    # Helper function to create a subsequent mask
    def subsequent_mask(self, size):
        """Mask out subsequent positions."""
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape, device=device), diagonal=1)
        return subsequent_mask == 0
    
