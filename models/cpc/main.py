from loader import load_audio_model, librispeech_loader
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--start_epoch", type=int, help="")
parser.add_argument("--num_epochs", type=int, help="")
parser.add_argument("--device", type=str, default="cuda:0", help="device")


def train(args, model, optimizer, writer):
    

    # get datasets and dataloaders
    train_loader, train_dataset, test_loader, test_dataset = librispeech_loader(
        args, num_workers=args.num_workers
    )
    model = load_audio_model(args)

    total_step = len(train_loader)
    best_loss = 0
    
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        loss_epoch = 0
        for step, (audio, filename, _, start_idx) in enumerate(train_loader):
            start_time = time.time()
            audio = audio.to(args.device)
            loss = model(audio)
            
            loss = loss.mean()
            
            