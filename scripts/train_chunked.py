"""Train the chunked latent-action model: `python scripts/train_chunked.py --run-name NAME` (torchrun for several GPUs)."""
from lawm.train.trainer import main

if __name__ == "__main__":
    main()
