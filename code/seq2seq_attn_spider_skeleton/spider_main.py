import heapq
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from model import Decoder, Encoder
from spider import DataLoader


class Beam:
    def __init__(self, score, seq, h, c):
        self.score = score
        self.seq = seq
        self.h = h
        self.c = c

    def __lt__(self, other):
        return self.score < other.score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader("train")

    num_layers = 2
    input_size = len(data_loader.vocabs_q)
    hidden_size = 250  # 512
    num_classes = len(data_loader.vocabs_f)

    encoder = Encoder(input_size, hidden_size, num_layers).to(device)
    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=0.01 * 0.95, alpha=0.95)
    decoder = Decoder(input_size, hidden_size, num_layers, num_classes).to(device)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=0.01 * 0.95, alpha=0.95)

    criterion = nn.NLLLoss()
    num_epochs = 20 * 80

    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0
        batch = data_loader.random_batch()
        for i, (lang, logic) in enumerate(batch):
            lang = lang.to(device)
            logic = logic.to(device)

            enc_out, h, c = encoder(lang)
            torch.tensor([data_loader.get_start_idx()]).to(device)

            for j in range(1, len(logic)):
                pred, h, c = decoder(logic[:j], h, c, enc_out)
                loss += criterion(pred, logic[1 : j + 1])

        loss /= len(batch)
        loss.backward()
        torch.nn.utils.clip_grad_value_(encoder.parameters(), 5)
        torch.nn.utils.clip_grad_value_(decoder.parameters(), 5)

        encoder_optimizer.step()
        decoder_optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item())
            )
            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, loss.item()
                ),
                file=sys.stderr,
            )
            loss = 0
            for lang, logic in data_loader.validation_data:
                lang = lang.to(device)
                logic = logic.to(device)

                enc_out, h, c = encoder(lang)
                torch.tensor([data_loader.get_start_idx()]).to(device)
                for j in range(1, len(logic)):
                    pred, h, c = decoder(logic[:j], h, c, enc_out)
                    loss += criterion(pred, logic[1 : j + 1])
                    # out = pred.max(1)[1]
            loss /= len(data_loader.validation_data)
            print("Validation Loss: {:.4f}".format(loss.item()))
            print("Validation Loss: {:.4f}".format(loss.item()), file=sys.stderr)

    encoder.eval()
    decoder.eval()
    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")

    test_data = DataLoader("test").data
    max_decode_len = max([len(logic) for _, logic, _ in test_data])
    beam_width = 5

    for i, (lang, logic, db) in enumerate(test_data):
        lang = lang.to(device)
        logic = logic.to(device)

        enc_out, h, c = encoder(lang)
        start_token = data_loader.get_start_idx()
        end_token = data_loader.get_end_idx()

        beams = [Beam(0, [start_token], h, c)]

        for _ in range(max_decode_len):
            new_beams = []

            for beam in beams:
                if beam.seq[-1] == end_token:
                    new_beams.append(beam)
                    continue

                seq = torch.tensor(beam.seq).to(device)
                pred, h, c = decoder(seq, beam.h, beam.c, enc_out)
                for idx, log_prob in enumerate(pred[-1]):
                    new_seq = beam.seq + [idx]
                    new_score = beam.score + log_prob.item()
                    new_beams.append(Beam(new_score, new_seq, h, c))

            beams = heapq.nlargest(beam_width, new_beams)

            if all(beam.seq[-1] == end_token for beam in beams):
                break

        best_beam = max(beams, key=lambda b: b.score)
        decoded_seq = [data_loader.vocabs_f[idx] for idx in best_beam.seq[1:]]
        print(" ".join(decoded_seq))


if __name__ == "__main__":
    main()
