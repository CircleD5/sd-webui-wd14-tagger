import os
import pandas as pd
import numpy as np

from typing import Tuple, List, Dict
from io import BytesIO
from PIL import Image

from pathlib import Path
from huggingface_hub import hf_hub_download

from modules import shared
import re

tag_escape_pattern = re.compile(r'([\\()])')

# i'm not sure if it's okay to add this file to the repository
from . import dbimutils

# select a device to process
_use_cpu = getattr(shared.cmd_opts, "use_cpu", []) or []
use_cpu = ('all' in _use_cpu) or ('interrogate' in _use_cpu)


class Interrogator:
    @staticmethod
    def postprocess_tags(
            tags: Dict[str, float],
            threshold=0.35,
            additional_tags: List[str] = [],
            exclude_tags: List[str] = [],
            sort_by_alphabetical_order=False,
            add_confident_as_weight=False,
            replace_underscore=False,
            replace_underscore_excludes: List[str] = [],
            escape_tag=False
    ) -> Dict[str, float]:
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c

            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # filter tags
            if (
                    c >= threshold
                    and t not in exclude_tags
            )
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags

    def __init__(self, name: str) -> None:
        self.name = name

    def load(self):
        raise NotImplementedError()

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            print(f'Unloaded {self.name}')

        if hasattr(self, 'tags'):
            del self.tags

        return unloaded

    def interrogate(
            self,
            image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        raise NotImplementedError()


class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
            self,
            name: str,
            model_path='model.onnx',
            tags_path='selected_tags.csv',
            **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")

        model_path = Path(hf_hub_download(
            **self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            **self.kwargs, filename=self.tags_path))
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()

        # only one of these packages should be installed at a time in any one environment
        # https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime
        # TODO: remove old package when the environment changes?
        from launch import is_installed, run_pip
        if not is_installed('onnxruntime'):
            package = os.environ.get(
                'ONNXRUNTIME_PACKAGE',
                'onnxruntime-gpu'
            )

            run_pip(f'install {package}', 'onnxruntime')

        from onnxruntime import InferenceSession

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers.pop(0)

        self.model = InferenceSession(str(model_path), providers=providers)

        print(f'Loaded {self.name} model from {model_path}')

        self.tags = pd.read_csv(tags_path)

    def interrogate(
            self,
            image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][['name']]
        tags['confidents'] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        return ratings, tags
class OppaiInterrogator(Interrogator):
    def __init__(
            self,
            name: str,
            model_path='model.onnx',
            tags_path='selected_tags.csv',
            preproc_path='preprocessing.json',
            **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.preproc_path = preproc_path
        self.kwargs = kwargs

    def download(self) -> Tuple[os.PathLike, os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")
        model_path = Path(hf_hub_download(**self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(**self.kwargs, filename=self.tags_path))
        preproc_path = Path(hf_hub_download(**self.kwargs, filename=self.preproc_path))
        return model_path, tags_path, preproc_path

    def load(self) -> None:
        model_path, tags_path, preproc_path = self.download()
        
        from launch import is_installed, run_pip
        if not is_installed('onnxruntime'):
            package = os.environ.get('ONNXRUNTIME_PACKAGE', 'onnxruntime-gpu')
            run_pip(f'install {package}', 'onnxruntime')

        from onnxruntime import InferenceSession
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers.pop(0)

        self.model = InferenceSession(str(model_path), providers=providers)
        print(f'Loaded {self.name} model from {model_path}')

        import csv
        import json
        self.tag_names = []
        self.categories = []
        with open(tags_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.tag_names.append(row["name"])
                self.categories.append(int(row["category"]))
        
        self.skip_mask = np.zeros(len(self.tag_names), dtype=bool)
        for i, name in enumerate(self.tag_names):
            if name in ("<PAD>", "<UNK>"):
                self.skip_mask[i] = True

        with open(preproc_path, encoding="utf-8") as f:
            preproc = json.load(f)
        self.image_size = int(preproc["image_size"])
        self.pad_color = tuple(int(c) for c in preproc["pad_color_rgb"])
        self.mean = np.array(preproc["normalize_mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(preproc["normalize_std"], dtype=np.float32).reshape(3, 1, 1)

    def interrogate(self, image: Image) -> Tuple[Dict[str, float], Dict[str, float]]:
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        img = image.convert("RGB")
        w, h = img.size
        size = self.image_size
        scale = min(size / w, size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (size, size), self.pad_color)
        x0 = (size - nw) // 2
        y0 = (size - nh) // 2
        canvas.paste(resized, (x0, y0))
        mask = np.ones((size, size), dtype=bool)
        mask[y0:y0 + nh, x0:x0 + nw] = False

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        arr = (arr - self.mean) / self.std
        pixel_values = arr.astype(np.float32)

        outputs = self.model.run(["probabilities"], {
            "pixel_values": pixel_values[None, ...],
            "padding_mask": mask[None, ...],
        })
        probs = outputs[0][0].astype(np.float32)

        tags = {}
        for i, p in enumerate(probs):
            if not self.skip_mask[i] and p > 0:
                tags[self.tag_names[i]] = float(p)
                
        return {}, tags