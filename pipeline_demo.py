"""Minimal demonstration pipeline for the LLM + classifier + UNet + refine agent flow.

This script keeps the components lightweight and dependency-free so the
end-to-end flow can run locally without heavy model downloads.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


# --- Components --------------------------------------------------------------

@dataclass
class LLMAnalyzer:
    """Very small stub that mimics extracting semantic cues from text.

    The analyzer builds a deterministic vector from characters so that the rest
    of the pipeline can consume a repeatable signal.
    """

    embedding_dim: int = 8

    def __call__(self, text: str) -> List[float]:
        seed = sum(ord(ch) for ch in text)
        rng = random.Random(seed)
        vector = [rng.random() for _ in range(self.embedding_dim)]
        print(f"[LLM] embedding: {vector}")
        return vector


@dataclass
class LanguageClassifier:
    """Tiny heuristic classifier that tags the input language.

    It is intentionally simplistic—production code would swap in a trained
    classifier. The output is used to pick the next actor in the pipeline.
    """

    known_languages: Tuple[str, ...] = ("en", "zh")

    def __call__(self, text: str) -> str:
        contains_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in text)
        label = "zh" if contains_chinese else "en"
        print(f"[Classifier] detected language: {label}")
        return label


@dataclass
class TinyUNet:
    """Toy UNet-inspired block that produces a coarse mask.

    The implementation runs a couple of lightweight convolutions in pure
    Python to avoid pulling in deep learning frameworks. The network ingests a
    2D image (grayscale) and emits a probability mask in [0, 1].
    """

    kernel: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 2.0, 1.0),
        (2.0, 4.0, 2.0),
        (1.0, 2.0, 1.0),
    )

    def _get(self, grid: List[List[float]], i: int, j: int) -> float:
        i = min(max(i, 0), len(grid) - 1)
        j = min(max(j, 0), len(grid[0]) - 1)
        return grid[i][j]

    def _convolve(self, image: List[List[float]]) -> List[List[float]]:
        output = [[0.0 for _ in row] for row in image]
        norm = sum(sum(row) for row in self.kernel)
        for i in range(len(image)):
            for j in range(len(image[0])):
                acc = 0.0
                for ki in range(3):
                    for kj in range(3):
                        acc += self.kernel[ki][kj] * self._get(image, i + ki - 1, j + kj - 1)
                output[i][j] = acc / norm
        return output

    def __call__(self, image: List[List[float]]) -> List[List[float]]:
        smoothed = self._convolve(image)
        flat = [v for row in smoothed for v in row]
        lo, hi = min(flat), max(flat)
        ptp = hi - lo if hi != lo else 1e-6
        normalized = [[(v - lo) / ptp for v in row] for row in smoothed]
        print(
            f"[UNet] coarse mask stats: min={min(flat):.3f}, max={max(flat):.3f}, range={ptp:.3f}"
        )
        return normalized


@dataclass
class RefineAgent:
    """Simple refiner that sharpens the coarse mask using text context.

    The embedding is converted into a scaling factor that controls how
    aggressively the mask is sharpened. The refinement is a single power-law
    transform followed by thresholding.
    """

    threshold: float = 0.5

    def __call__(self, mask: List[List[float]], text_embedding: List[float]) -> List[List[float]]:
        scale = 0.8 + 0.4 * (sum(text_embedding) / len(text_embedding))
        sharpened = [[max(0.0, min(1.0, v ** scale)) for v in row] for row in mask]
        refined = [[1.0 if v > self.threshold else 0.0 for v in row] for row in sharpened]
        coverage = sum(v for row in refined for v in row) / (len(refined) * len(refined[0]))
        print(
            f"[Refine] scale={scale:.3f}, final coverage={coverage:.3f} (fraction of pixels > threshold)"
        )
        return refined


# --- Orchestration -----------------------------------------------------------

@dataclass
class PipelineResult:
    language: str
    embedding: List[float]
    coarse_mask: List[List[float]]
    refined_mask: List[List[float]]


def run_pipeline(text: str, image: List[List[float]]) -> PipelineResult:
    llm = LLMAnalyzer()
    classifier = LanguageClassifier()
    unet = TinyUNet()
    refine = RefineAgent()

    embedding = llm(text)
    language = classifier(text)
    coarse_mask = unet(image)
    refined_mask = refine(coarse_mask, embedding)
    return PipelineResult(language, embedding, coarse_mask, refined_mask)


# --- Demo --------------------------------------------------------------------

def demo() -> None:
    text = "在血管区域进行分割"
    size = 32
    coords = [i * 2.0 / (size - 1) - 1.0 for i in range(size)]
    image = []
    for y in coords:
        row = []
        for x in coords:
            val = math.exp(-4 * (x ** 2 + y ** 2))
            row.append(val)
        image.append(row)

    flat = [v for row in image for v in row]
    lo, hi = min(flat), max(flat)
    ptp = hi - lo if hi != lo else 1e-6
    image = [[(v - lo) / ptp for v in row] for row in image]

    print("Running pipeline demo...\n")
    result = run_pipeline(text, image)

    print("\nSummary")
    print(f"language: {result.language}")
    print(f"embedding (first 3 dims): {result.embedding[:3]}")
    flat_coarse = [v for row in result.coarse_mask for v in row]
    flat_refined = [v for row in result.refined_mask for v in row]
    print(f"coarse mask mean: {sum(flat_coarse) / len(flat_coarse):.3f}")
    print(f"refined mask coverage: {sum(flat_refined) / len(flat_refined):.3f}")


if __name__ == "__main__":
    demo()
