# AgentUniVesSeg
This is our work on Multi-agent Universal Vessel Segmentation.

## Minimal pipeline demo

`pipeline_demo.py` provides a lightweight, dependency-free demonstration of the
LLM → classifier → Tiny UNet → refine agent flow sketched in the design. To run
the example end-to-end:

```bash
python pipeline_demo.py
```

The script prints the detected language, the synthesized text embedding, and
basic statistics for both the coarse and refined segmentation masks.
