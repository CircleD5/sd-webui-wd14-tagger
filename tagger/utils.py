from typing import List, Dict
from pathlib import Path

from modules import scripts
from tagger.preset import Preset
from tagger.interrogator import Interrogator, WaifuDiffusionInterrogator, OppaiInterrogator

preset = Preset(Path(scripts.basedir(), 'presets'))

interrogators: Dict[str, Interrogator] = {}


def refresh_interrogators() -> List[str]:
    global interrogators
    interrogators = {
        'wd-eva02-large-tagger-v3': WaifuDiffusionInterrogator(
            'wd-eva02-large-tagger-v3',
            repo_id='SmilingWolf/wd-eva02-large-tagger-v3',
        ),
        'wd-vit-large-tagger-v3': WaifuDiffusionInterrogator(
            'wd-vit-large-tagger-v3',
            repo_id='SmilingWolf/wd-vit-large-tagger-v3',
        ),
        'wd-vit-v3': WaifuDiffusionInterrogator(
            'wd14-vit-v3',
            repo_id='SmilingWolf/wd-vit-tagger-v3',
        ),
        'wd-swinv2-v3': WaifuDiffusionInterrogator(
            'wd-swinv2-v3',
            repo_id='SmilingWolf/wd-swinv2-tagger-v3',
        ),
        'wd-convnext-v3': WaifuDiffusionInterrogator(
            'wd-convnext-v3',
            repo_id='SmilingWolf/wd-convnext-tagger-v3',
        ),
        'wd14-convnextv2-v2': WaifuDiffusionInterrogator(
            'wd14-convnextv2-v2',
            repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
            revision='v2.0'
        ),
        'wd14-moat-v2': WaifuDiffusionInterrogator(
            'wd-v1-4-moat-tagger-v2',
            repo_id='SmilingWolf/wd-v1-4-moat-tagger-v2',
            revision='v2.0'
        ),
        'wd14-vit-v2': WaifuDiffusionInterrogator(
            'wd14-vit-v2',
            repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2',
            revision='v2.0'
        ),
        'wd14-convnext-v2': WaifuDiffusionInterrogator(
            'wd14-convnext-v2',
            repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2',
            revision='v2.0'
        ),
        'wd14-swinv2-v2': WaifuDiffusionInterrogator(
            'wd14-swinv2-v2',
            repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2',
            revision='v2.0'
        ),
        'wd14-convnextv2-v2-git': WaifuDiffusionInterrogator(
            'wd14-convnextv2-v2',
            repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
        ),
        'wd14-vit-v2-git': WaifuDiffusionInterrogator(
            'wd14-vit-v2-git',
            repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2'
        ),
        'wd14-convnext-v2-git': WaifuDiffusionInterrogator(
            'wd14-convnext-v2-git',
            repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2'
        ),
        'wd14-swinv2-v2-git': WaifuDiffusionInterrogator(
            'wd14-swinv2-v2-git',
            repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2'
        ),
        'wd14-vit': WaifuDiffusionInterrogator(
            'wd14-vit',
            repo_id='SmilingWolf/wd-v1-4-vit-tagger'),
        'wd14-convnext': WaifuDiffusionInterrogator(
            'wd14-convnext',
            repo_id='SmilingWolf/wd-v1-4-convnext-tagger'
        ),
        'oppai-onnx-v1.1': OppaiInterrogator(
            'oppai-onnx-v1.1',
            repo_id='Grio43/OppaiOracle',
            model_path='V1.1_onnx/model.onnx',
            tags_path='V1.1_onnx/selected_tags.csv',
            preproc_path='V1.1_onnx/preprocessing.json'
        ),
        'oppai-onnx-v1': OppaiInterrogator(
            'oppai-onnx-v1',
            repo_id='Grio43/OppaiOracle',
            model_path='V1_onnx/model.onnx',
            tags_path='V1_onnx/selected_tags.csv',
            preproc_path='V1_onnx/preprocessing.json'
        ),
    }

    return sorted(interrogators.keys())


def split_str(s: str, separator=',') -> List[str]:
    return [x.strip() for x in s.split(separator) if x]
