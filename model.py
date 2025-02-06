import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from config import block_size, vocab_size, stoi


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=16, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(context)
            .view(B, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(context)
            .view(B, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, emb_size, encoder_emb_size, num_heads=16, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.cross_attn = CrossAttention(emb_size, encoder_emb_size, num_heads, dropout)
        self.feed_forward = FeedForward(emb_size, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        x = x + self.dropout(
            self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        )
        x = x + self.dropout(self.cross_attn(self.norm2(x), context))
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v2(weights="DEFAULT")
        self.model.classifier = nn.LayerNorm(1280)

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        emb_size=128,
        encoder_emb_size=1280,
        num_heads=16,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(block_size, emb_size)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(emb_size, encoder_emb_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, vocab_size)

    def forward(self, x, context):
        Bx, Tx = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(Tx, device=x.device))
        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x, context)

        x = self.norm(x)
        logits = self.linear(x)
        return logits


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        emb_size=128,
        encoder_emb_size=1280,
        num_heads=16,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(
            emb_size, encoder_emb_size, num_heads, num_layers, dropout
        )

    def forward(self, x, im):
        context = self.encoder(im)
        logits = self.decoder(x, context)
        return logits

    @torch.no_grad()
    def generate(self, im, max_new_tokens=10):
        context = self.encoder(im)
        idx = torch.zeros((1, 1), dtype=torch.long, device=im.device)
        probs_log = [1.0]
        for _ in range(max_new_tokens):
            logits = self.decoder(idx, context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            prob_next = probs.max(dim=-1).values.item()
            probs_log.append(prob_next)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == stoi[">"]:
                break

        idx = idx.tolist()

        return idx[0], probs_log