# Fashion-MNIST CVAE vs DDPM

Class-conditional image generation on Fashion-MNIST: a CVAE and a DDPM (UNet + DDPM scheduler).

| Notebook | What |
| --- | --- |
| `cvae_fashion_mnist.ipynb` | Train / sample CVAE |
| `stable_diffusion_fashion_mnist.ipynb` | Train / sample DDPM |
| `compare_fashion_mnist.ipynb` | Benchmark (loads checkpoints only) |
| `visualize_samples_fashion_mnist.ipynb` | Sample grids + DDPM denoise GIFs |

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision matplotlib diffusers accelerate jupyter
jupyter notebook
```

Train CVAE and DDPM first (or set `RUN_TRAINING = False` if checkpoints already exist). Then run compare / visualize.

Checkpoints: `./checkpoints/cvae_fashion_mnist.pt`, `./checkpoints/ddpm_fashion_mnist/`, `./checkpoints/fashion_mnist_classifier.pt`.

## Comparison & benchmark

Same protocol for both models: 200 samples/class, shared seeds, images in `[0, 1]`.

- **Conditional accuracy** — pretrained Fashion-MNIST classifier; fraction of generated images labeled as the intended class.
- **Feature FID** — Fréchet distance between real and generated features from that classifier (not Inception).
- **Time** — wall-clock seconds per image.

Last run: DDPM wins quality (acc 0.83 vs 0.63, FID 41 vs 124); CVAE is much faster.
