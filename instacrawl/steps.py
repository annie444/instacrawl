"""Dataclasses for the Instagram scraper."""

from dataclasses import dataclass
from os import PathLike
from rich.progress import Progress, TaskID
from typing import Any, Dict, List, Literal, Iterator, Tuple
from collections.abc import MutableMapping
import numpy as np
import os
from PIL import Image
from decord import VideoReader, cpu
from pathlib import Path
import face_recognition


@dataclass
class Post:
    """Class to store post data."""

    url: str
    file: PathLike


@dataclass
class DownloadRequest:
    """Class to store download request data."""

    path: PathLike
    post: List[Post]


@dataclass
class InstaStepOne:
    """Step one of the Instagram scraper."""

    profile_page: str
    username: str
    password: str
    cookies: Dict[str, str]
    path: PathLike
    hrefs: List[str]
    alts: List[str]
    srcs: List[str]
    articles: List[str]
    contents: List[str]
    dates: List[str]
    times: List[str]
    requests: List[str]
    items: List[Any]
    likes: List[int]
    comments: List[int]
    posts: List[str]


@dataclass
class InstaStepTwo:
    """Step two of the Instagram scraper."""

    download_requests: List[DownloadRequest]
    cookies: Dict[str, str]
    path: PathLike
    hrefs: List[str]
    paths: List[List[PathLike | str]]
    types: List[Literal["video", "image", ""]]


@dataclass
class InstaStepThree:
    """Step three of the Instagram scraper."""

    hrefs: List[str]
    alts: List[str]
    srcs: List[str]
    articles: List[str]
    contents: List[str]
    dates: List[str]
    times: List[str]
    items: List[Any]
    comments: List[int]
    likes: List[int]
    posts: List[str]
    paths: List[List[PathLike | str]]
    types: List[Literal["video", "image", ""]]


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def get_video(path: PathLike) -> np.ndarray:
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    vr.seek(0)
    indicies = sample_frame_indices(
        clip_len=16,
        frame_sample_rate=4,
        seg_len=len(vr)
    )
    video = vr.get_batch(indicies).asnumpy()
    return video


def get_image(path: PathLike) -> Tuple[Image.Image, np.ndarray]:
    image = Image.open(Path(path))
    array = face_recognition.load_image_file(path)
    return image, array


class VideoData:
    post: int
    index: int
    path: PathLike
    video: np.ndarray

    def __init__(
        self,
        post: int,
        index: int,
        path: PathLike,
        video: np.ndarray,
    ):
        self.post = post
        self.index = index
        self.path = path
        self.video = video


class PersonImage:
    post: int
    image: int
    index: int
    path: PathLike
    person: Image.Image

    def __init__(
        self,
        post: int,
        image: int,
        index: int,
        path: PathLike,
        person: Image.Image,
    ):
        self.post = post
        self.image = image
        self.index = index
        self.path = path
        self.person = person


@dataclass
class FaceData:
    post: int
    image: int
    path: PathLike
    encoding: np.ndarray
    box: Tuple[
        int,
        Any,
        Any,
        int
    ]


class ImageData:
    post: int
    index: int
    path: PathLike
    image: Image.Image
    array: np.ndarray

    def __init__(
        self,
        post: int,
        index: int,
        path: PathLike,
        image: Image.Image,
        array: np.ndarray,
    ):
        self.post = post
        self.index = index
        self.path = path
        self.image = image
        self.array = array


class InstaPost:
    """Class to store post data."""

    href: str
    alt: str
    src: str
    article: str
    content: str
    date: str
    time: str
    item: Any
    comments: int
    likes: int
    post: str
    path: List[PathLike]
    type_: Literal["video", "image", ""]
    videos: List[PathLike]
    images: List[PathLike]

    def __init__(
        self,
        href: str,
        alt: str,
        src: str,
        article: str,
        content: str,
        date: str,
        time: str,
        item: Any,
        comments: int,
        likes: int,
        post: str,
        path: List[PathLike],
        type_: Literal["video", "image", ""],
    ):
        self.href = href
        self.alt = alt
        self.src = src
        self.article = article
        self.content = content
        self.date = date
        self.time = time
        self.item = item
        self.comments = comments
        self.likes = likes
        self.post = post
        self.path = path
        self.type_ = type_

        self._media()

    def _media(self):
        self.videos = []
        self.images = []
        for path in self.path:
            if str(path).endswith(".mp4"):
                if os.path.exists(path):
                    self.videos.append(path)
            elif str(path).endswith(".jpg"):
                if os.path.exists(path):
                    self.images.append(path)


class InstaStepFour(MutableMapping):
    """Step four of the Instagram scraper."""

    def __init__(
        self,
        url: str,
        path: PathLike,
        username: str,
        password: str,
        posts: List[InstaPost],
        progress: Progress | None = None,
        task: TaskID | None = None,
    ):
        self.url: str = url
        self.path: PathLike = path
        self.username: str = username
        self.password: str = password
        self.posts: Dict[int, InstaPost] = {}
        self.videos: List[VideoData] = []
        self.images: List[ImageData] = []
        self.people: List[PersonImage] = []
        self.faces: List[FaceData] = []
        for i, post in enumerate(posts):
            self.posts[i] = post
            if progress is not None and task is not None:
                progress.update(task, advance=1)
            for j, video in enumerate(post.videos):
                self.videos.append(
                    VideoData(
                        post=i,
                        index=j,
                        path=video,
                        video=get_video(video)
                    )
                )
            for j, image in enumerate(post.images):
                im, ar = get_image(image)
                self.images.append(
                    ImageData(
                        post=i,
                        index=j,
                        path=image,
                        image=im,
                        array=ar,
                    )
                )

    @property
    def encodings(self) -> List[np.ndarray]:
        encodings: List[np.ndarray] = []
        for face in self.faces:
            encodings.append(face.encoding)
        return encodings

    def __len__(self) -> int:
        """Return the number of posts."""
        return len(self.posts)

    def __iter__(self) -> Iterator[InstaPost]:
        """Return an iterator."""
        return iter(self.posts)

    def __getitem__(self, index: int) -> InstaPost:
        """Return a post."""
        return self.posts[index]

    def __setitem__(self, index: int, item: InstaPost) -> None:
        self.posts[index] = item

    def __delitem__(self, index: int) -> None:
        del self.posts[index]

    def __contains__(self, index: int) -> bool:
        return index in self.posts.keys()

    def keys(self) -> List[int]:
        return list(self.posts.keys())

    def values(self) -> List[InstaPost]:
        return list(self.posts.values())

    def items(self) -> List[Tuple[int, InstaPost]]:
        return list(self.posts.items())
