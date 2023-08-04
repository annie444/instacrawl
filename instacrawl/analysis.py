"""Objects to analyze the crawled data."""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
from transformers import \
    AutoImageProcessor, \
    AutoModelForObjectDetection, \
    AutoTokenizer, \
    AutoConfig, \
    RobertaForSequenceClassification, \
    AutoModelForSequenceClassification, \
    AutoModelForVideoClassification, \
    AutoModelForSemanticSegmentation
import transformers
import numpy as np
import pandas as pd
from PIL import Image
import torch
from rich.progress import Progress, TaskID
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import DBSCAN
import face_recognition
from instacrawl.steps import InstaStepFour, PersonImage, FaceData
import time as sleep
from instacrawl.console import console
import os

np.random.seed(0)
transformers.logging.set_verbosity(transformers.logging.CRITICAL)


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


class Postalyzer:
    """Analyze the crawled data."""

    def __init__(
        self,
        data: InstaStepFour
    ):
        """Initialize the analysis."""
        self.data = data
        self.nlp = TextAnalysis(self.data)
        self.vid = VideoAnalysis(self.data)
        self.img = ImageAnalysis(self.data)

    def analyze(self) -> List[pd.DataFrame]:
        data: List[pd.DataFrame] = []
        with Progress(console=console) as progress:
            self.progress = progress
            task = self.progress.add_task(
                "Analyzing posts...",
                total=None
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = []
                f = executor.submit(
                    self.vid.run,
                    self.progress
                )
                futures.append(f)
                f = executor.submit(
                    self.nlp.run,
                    self.progress
                )
                futures.append(f)
                f = executor.submit(
                    self.img.run,
                    self.progress
                )
                futures.append(f)
                for fut in as_completed(futures):
                    exce = fut.exception()
                    if exce is not None:
                        console.log(exce)
                    else:
                        data.append(fut.result())
            self.progress.update(task, completed=1)
        return data


class InstaAnalysis(object):
    def __init__(
        self,
        data: InstaStepFour,
        name: str
    ):
        self.data = data
        self.task: TaskID | None = None
        self.completed: int | None = None
        self.results: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.name = name

    def insert_result(
        self,
        post: int,
        cat: str,
        label: str,
        data: Any,
        append: bool = False
    ):
        if post not in self.results.keys():
            self.results[post] = {}
        if cat not in self.results[post].keys():
            self.results[post][cat] = {}
        if append:
            if label not in self.results[post][cat].keys():
                self.results[post][cat][label] = [data]
            else:
                self.results[post][cat][label].append(data)
        else:
            self.results[post][cat][label] = data

    def analyze(self):
        ...

    def run(
        self,
        progress: Progress | None = None,
    ) -> pd.DataFrame:
        self.progress = progress
        self.analyze()
        sleep.sleep(3)
        data = self.save()
        return data

    def track(self, name: str, len: int | None = None):
        if hasattr(self, "progress") and isinstance(
            self.progress,
            Progress
        ):
            if not name.endswith("..."):
                name = name + "..."
            self.task = self.progress.add_task(
                name,
                total=len,
            )
            self.completed = len if len is not None else 1

    def advance(self):
        if (hasattr(self, "progress") and
                hasattr(self, "task") and
                isinstance(
            self.progress,
            Progress
        ) and isinstance(
            self.task,
            int
        )):
            self.progress.update(self.task, advance=1)

    def complete(self):
        if (hasattr(self, "progress") and
                hasattr(self, "task") and
                hasattr(self, "completed") and
                isinstance(
            self.progress,
            Progress
        ) and isinstance(
            self.task,
            int
        ) and isinstance(
            self.completed,
            int,
        )):
            self.progress.update(self.task, completed=self.completed)
            self.task = None
            self.completed = None

    def save(self) -> pd.DataFrame:
        self.track(
            f"Saving the {self.name} results",
            ((len(self.results) * 2) + 2)
        )

        index: List[int] = []
        columns: List[Tuple[str, str]] = []
        for key in self.results.keys():
            if key not in index:
                index.append(key)
            for col1_key in self.results[key].keys():
                for col2_key in self.results[key][col1_key].keys():
                    if (col1_key, col2_key) not in columns:
                        columns.append((col1_key, col2_key))
            self.advance()

        lists: List[List[Any]] = []
        for key in index:
            row: List[Any] = []
            for (col1_key, col2_key) in columns:
                if col1_key not in self.results[key].keys():
                    row.append(None)
                elif col2_key not in self.results[key][col1_key].keys():
                    row.append(None)
                else:
                    row.append(self.results[key][col1_key][col2_key])
            lists.append(row)
            self.advance()

        data = pd.DataFrame(
            lists,
            columns=pd.MultiIndex.from_tuples(columns),
            index=index
        )
        self.advance()
        data.to_html(
            os.path.join(
                self.data.path,
                f"{self.name}_analysis.html"
            )
        )
        data.to_excel(
            os.path.join(
                self.data.path,
                f"{self.name}_analysis.xlsx"
            )
        )
        data.to_pickle(
            os.path.join(
                self.data.path,
                f"{self.name}_analysis.pkl"
            )
        )
        self.complete()
        return data


# Results structure:
#   self.results: Dict[int, Dict[str, Dict[str, Any]]] = {
#       PostIndex: {
#           measurementCategory: {
#               measurementName: measurementData
#           }
#       }
#   }


class VideoAnalysis(InstaAnalysis):
    """Analyze the crawled videos."""

    def __init__(
        self,
        data: InstaStepFour,
        model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
    ):
        """Initialize the analysis."""
        super().__init__(data=data, name="video")
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVideoClassification.from_pretrained(
            self.model_name
        )

    def analyze(self):
        """Analyze the videos."""
        self.track("Analyzing videos", len(self.data.videos))
        for video in self.data.videos:
            self.inputs = self.processor(
                list(video.video),
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model(**self.inputs)
                logits = output.logits
            label = logits.argmax(-1).item()
            self.insert_result(
                video.post,
                "activity",
                self.model.config.id2label[label],
                logits[0][label].item()
            )
            self.advance()
        self.complete()


class ImageAnalysis(InstaAnalysis):
    """Analyze the crawled images."""

    def __init__(
        self,
        data: InstaStepFour,
        model_name: str = "facebook/detr-resnet-50",
        transformer_name: str = "mattmdjaga/segformer_b2_clothes"
    ):
        """Initialize the analysis."""
        super().__init__(
            data=data,
            name="image"
        )
        self.model_name = model_name
        self.transformer_name = transformer_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_name
        )
        self.extractor = AutoImageProcessor.from_pretrained(
            self.transformer_name
        )
        self.transformer = \
            AutoModelForSemanticSegmentation.from_pretrained(
                self.transformer_name
            )
        self.annotations: Dict[int, str] = {
            1: "hat",
            2: "hair",
            3: "sunglasses",
            4: "upperclothes",
            5: "skirt",
            6: "pants",
            7: "dress",
            8: "belt",
            9: "leftShoe",
            10: "rightShoe",
            11: "face",
            12: "leftLeg",
            13: "rightLeg",
            14: "leftArm",
            15: "rightArm",
            16: "bag",
            17: "scarf"
        }

    def analyze(self):
        """Analyze the videos."""
        self.detector()
        self.segmenter()
        self.recognizer()
        self.classifier()
        sleep.sleep(3)

    def detector(self):
        self.track("Detecting objects", len(self.data.images))
        for image in self.data.images:
            inputs = self.processor(
                images=image.image,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                target_sizes = torch.tensor([image.image.size[::-1]])
                outs = self.processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=0.94
                )[0]
                boxes = outs['boxes'].detach().numpy()
                labels = outs['labels'].detach().numpy()
                scores = outs['scores'].detach().numpy()
            p = 0
            for score, label, box in zip(
                scores,
                labels,
                boxes
            ):
                label = self.model.config.id2label[label.item()]
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                self.insert_result(image.post, label, "score", score.item())
                self.insert_result(image.post, label, "xmin", xmin)
                self.insert_result(image.post, label, "ymin", ymin)
                self.insert_result(image.post, label, "xmax", xmax)
                self.insert_result(image.post, label, "ymax", ymax)
                if label == "person":
                    self.insert_result(image.post, "people", "count", p)
                    x1 = 0 if int(xmin) < 0 else int(xmin)
                    x2 = int(xmax)
                    y1 = 0 if int(ymin) < 0 else int(ymin)
                    y2 = int(ymax)
                    self.data.people.append(
                        PersonImage(
                            post=image.post,
                            image=image.index,
                            index=p,
                            path=image.path,
                            person=Image.fromarray(
                                np.array(image.image)[y1:y2, x1:x2]
                            )
                        )
                    )
                    p += 1
            self.advance()
        self.complete()

    def segmenter(self):
        self.track(
            "Segmenting people",
            (
                len(self.data.people) *
                len(self.annotations)
            )
        )
        for person in self.data.people:
            if person.post not in self.results.keys():
                self.results[person.post] = {}
            if "visible" not in self.results[person.post].keys():
                self.results[person.post]["visible"] = {
                    "hat": 0,
                    "hair": 0,
                    "sunglasses": 0,
                    "upperclothes": 0,
                    "skirt": 0,
                    "pants": 0,
                    "dress": 0,
                    "belt": 0,
                    "leftShoe": 0,
                    "rightShoe": 0,
                    "face": 0,
                    "leftLeg": 0,
                    "rightLeg": 0,
                    "leftArm": 0,
                    "rightArm": 0,
                    "bag": 0,
                    "scarf": 0
                }
            inputs = self.extractor(images=person.person, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
            with torch.no_grad():
                outputs = self.transformer(pixel_values=pixel_values)
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=person.person.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            for num, label in self.annotations.items():
                segment = pred_seg.clone()
                self.advance()
                if num not in segment or 0 not in segment:
                    continue
                self.results[person.post]["visible"][label] += 1
        self.complete()

    def recognizer(self):
        self.track("Recognizing faces", len(self.data.images))
        for face in self.data.images:
            boxes = face_recognition.face_locations(
                face.array,
                model='cnn'
            )
            encodings = face_recognition.face_encodings(
                face.array,
                boxes
            )
            for box, encoding in zip(boxes, encodings):
                self.data.faces.append(
                    FaceData(
                        post=face.post,
                        image=face.index,
                        path=face.path,
                        encoding=encoding,
                        box=box,
                    )
                )
            self.insert_result(
                face.post,
                "faces",
                "quantity",
                len(boxes),
            )
            self.advance()
        self.complete()

    def classifier(self):
        self.track(
            "Identifying faces",
            3
        )
        encodings = [np.array(face.encoding) for face in self.data.faces]
        clt = DBSCAN(eps=0.5, metric="euclidean")
        self.advance()
        clt.fit(encodings)
        self.advance()
        for face, label in zip(self.data.faces, clt.labels_):
            self.insert_result(face.post, "faces", f"{label}", 1)
        self.complete()


class TextAnalysis(InstaAnalysis):
    def __init__(
        self,
        data: InstaStepFour,
        model_name: str = "SamLowe/roberta-base-go_emotions",
        classifier_name: str = "alimazhar-110/website_classification",
        ranking_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ):
        """Initialize the analysis."""
        super().__init__(
            data=data,
            name="text"
        )
        self.model_name = model_name
        self.classifier_name = classifier_name
        self.ranking_name = ranking_name
        self.rank_tokenizer = AutoTokenizer.from_pretrained(self.ranking_name)
        self.rank_config = AutoConfig.from_pretrained(self.ranking_name)
        self.rank_model = RobertaForSequenceClassification.from_pretrained(
            self.ranking_name,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.token_config = AutoConfig.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
        )

        self.classifier = AutoTokenizer.from_pretrained(self.classifier_name)
        self.class_config = AutoConfig.from_pretrained(self.classifier_name)
        self.class_model = AutoModelForSequenceClassification.from_pretrained(
            self.classifier_name
        )

    def analyze(self):
        self.rank()
        self.token()
        self.classify()

    def rank(self):
        self.track("Ranking the affect", len(self.data))
        for tag, post in self.data.posts.items():
            encoded_input = self.rank_tokenizer(
                post.article,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                output = self.rank_model(**encoded_input)
            scores = output[0][0].softmax(dim=0)
            scores = scores.detach().numpy()
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            self.advance()
            for i in range(scores.shape[0]):
                label = self.rank_config.id2label[ranking[i]]
                score = scores[ranking[i]]
                self.insert_result(tag, "positivity", label, score)
        self.complete()

    def classify(self):
        self.track("Classifying the text", len(self.data))
        for tag, post in self.data.posts.items():
            encoded_input = self.classifier(
                post.article,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                output = self.class_model(**encoded_input)
            scores = output[0][0].softmax(dim=0)
            scores = scores.detach().numpy()
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            self.advance()
            for i in range(scores.shape[0]):
                label = self.class_config.id2label[ranking[i]]
                score = scores[ranking[i]]
                self.insert_result(tag, "topics", label, score)
        self.complete()

    def token(self):
        self.track("Analyzing emotions", len(self.data))
        for tag, post in self.data.posts.items():
            encoded_input = self.tokenizer(
                post.article,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                output = self.model(**encoded_input)
            scores = output[0][0].softmax(dim=0)
            scores = scores.detach().numpy()
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            self.advance()
            for i in range(scores.shape[0]):
                label = self.token_config.id2label[ranking[i]]
                score = scores[ranking[i]]
                self.insert_result(tag, "emotions", label, score)
        self.complete()
