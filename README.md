# NED Model

Hey folks, been tinkering with state space models lately after reading that Mamba paper. Put together this implementation I'm calling NED. It's got some solid features like multi-head selective scanning, mixture of experts for scaling, optional hybrid attention, and Triton-optimized kernels. Nothing revolutionary, just a practical PyTorch take on efficient sequence modeling.

## Features

- Multi-head SSM with locally bidirectional scanning for improved context
- Sparse MoE integration with advanced routing
- Hybrid attention mechanisms (RoPE + ALiBi) in select layers
- Custom CUDA kernels for selective scan operations
- Quantization support via bitsandbytes
- Checkpoint conversion from NED SSM to PyTorch
- Comprehensive testing suite

The repo includes a sample `mamba-130m` model for quick testing.

## Installation

Clone the repo and set up:

```bash
git clone https://github.com/yourusername/ned_model.git
cd ned_model
pip install -r requirements.txt
python setup.py install
```

Note: For CUDA kernels, you might need to compile the C++ sources in `csrc/`. Make sure you have CUDA toolkit installed.

## Usage

Basic model loading and inference:

```python
from ned_model import NedForCausalLM, NedConfig
from transformers import AutoTokenizer

config = NedConfig.from_pretrained_model_name("ned-base")
model = NedForCausalLM(config)
tokenizer = AutoTokenizer.from_pretrained("mamba-130m")  # Assuming compatible tokenizer

input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=50)
print(tokenizer.decode(outputs[0]))
```

Check `test_comprehensive.py` for more detailed tests and examples.

## Contributing

Pull requests are welcome! Especially looking for help with:
- Training scripts and datasets
- Further optimizations (e.g., more Triton kernels)
- Integration with Hugging Face Hub
- Bug fixes in the custom ops

If you're interested in collaborating, shoot me a message on GitHub.

## Acknowledgments

This project builds on ideas from several papers:

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) by Albert Gu and Tri Dao
- Mixture of Experts concepts from [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) by Noam Shazeer et al.
- Additional inspirations from BlackMamba implementations

Big thanks to the authors and the open-source ML community for making this possible.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details. 
