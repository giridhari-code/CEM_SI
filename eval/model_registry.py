"""
CEM Eval: Model Registry & Gating.
Validates model performance and generates model cards.
"""

import os
import json
import datetime
from typing import Dict, Any

class ModelRegistry:
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)

    def check_gates(self, metrics: Dict[str, float], gates: Dict[str, float]) -> bool:
        """
        Checks if the model passes the minimum required thresholds.
        Gates example: {"accuracy": 0.50, "safety_score": 0.95}
        """
        passed = True
        failed_gates = []
        
        for metric_name, threshold in gates.items():
            actual_value = metrics.get(metric_name, 0.0)
            if actual_value < threshold:
                passed = False
                failed_gates.append(f"{metric_name}: {actual_value:.3f} < {threshold:.3f}")
                
        if not passed:
            print(f"❌ GATING FAILED: {failed_gates}")
        else:
            print("✅ GATING PASSED: Model is safe to deploy.")
            
        return passed

    def generate_model_card(self, model_name: str, metrics: Dict[str, float], config: Dict[str, Any]) -> str:
        """
        Generates a standard Model Card (Markdown).
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        card = f"""# Model Card: {model_name}

## Model Details
- **Architecture:** CEM (Cognitive Emergence Machine) - Looped Transformer-free
- **Date:** {date_str}
- **Base Config:** {config.get('base_config', 'N/A')}
- **Training Stages:** {config.get('stages', 'Pretrain -> SFT -> DPO')}

## Intended Use
This model is designed for text generation and reasoning tasks. It uses a recurrent depth loop (L3-L7) to adaptively compute.

## Evaluation Metrics
| Metric | Score | Gate Threshold | Status |
|--------|-------|----------------|--------|
"""
        for k, v in metrics.items():
            gate = config.get('gates', {}).get(k, 'N/A')
            status = "✅" if (isinstance(gate, float) and v >= gate) else "⚠️"
            if gate == 'N/A': status = "-"
            card += f"| {k} | {v:.4f} | {gate} | {status} |\n"
            
        card += """
## Limitations
- Character-level tokenizer (no BPE yet).
- Text-only (no multimodal audio/image in this version).
- May hallucinate on out-of-domain questions.
"""
        return card

    def register(self, model_name: str, metrics: Dict[str, float], config: Dict[str, Any], gates: Dict[str, float]):
        """Checks gates and saves the model card."""
        if not self.check_gates(metrics, gates):
            print(f"Model {model_name} rejected by registry.")
            return False
            
        # Save Model Card
        card_md = self.generate_model_card(model_name, metrics, {**config, "gates": gates})
        card_path = os.path.join(self.registry_dir, f"{model_name}_card.md")
        
        with open(card_path, "w") as f:
            f.write(card_md)
            
        # Save metadata JSON
        meta_path = os.path.join(self.registry_dir, f"{model_name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"metrics": metrics, "config": config}, f, indent=4)
            
        print(f"✅ Model {model_name} registered successfully. Card at {card_path}")
        return True